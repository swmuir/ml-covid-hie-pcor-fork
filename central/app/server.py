import gzip
import json
import os
import shutil
import signal
import time
from typing import Any,List
from queue import Queue
import threading

import centralConfigLoader
import db
import lightning as L
import torch
import torch.nn.functional as F
from debugPrint import Log, genDebugFunction
from fastapi import (
    FastAPI,
    HTTPException,
    Request,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

# from split_learning.models.vision.cnn_2d import CNN2D, CNN2DServer
from split_learning.models.bioServer import MainModel
from split_learning.models.mufasaCentral import MUFASA, InputEncoder, OutputDecoder
from split_learning.models.mufasaFull import MUFASAFull
from split_learning.schemas.message import MessageType, WSMessage
from split_learning.utils.serde import (
    decode_message_b64,
    deserialize_tensor,
    encode_message_b64,
    serialize_tensor,
)
from torch.utils.tensorboard import SummaryWriter
from uvicorn import Config, Server
import numpy as np
import mlflow
import mlflow.pytorch
import time
import logging
import s3fs
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler, FileSystemEvent, LoggingEventHandler
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
null = None
print = genDebugFunction(printTime=True)

start_time = time.time()

class StatePacket(BaseModel):
    state: Any = None


class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def send(self, websocket: WebSocket, message: Any):
        await websocket.send(message)

    async def send_bytes(self, websocket: WebSocket, message: bytes):
        await websocket.send_bytes(message)

    async def broadcast(self, message: Any):
        for connection in self.active_connections:
            await connection.send(message)

    async def broadcast_bytes(self, message: bytes):
        for connection in self.active_connections:
            await connection.send_bytes(message)




if "CONFIG" in os.environ:
    args = centralConfigLoader.CentralConfig(str(os.environ["CONFIG"]))
else:
    args = centralConfigLoader.CentralConfig("confCentral.py")

class SaveFileHandler(FileSystemEventHandler):
    '''
    Track create and modify events of files in /saves folder and put them in a queue.
    Consumers x 3 (range(1,4)) in threads will pick up from the queue and upload to s3 independence of main thread (training thread) .
    '''
    def __init__(self):
        self.s3 = s3fs.S3FileSystem()
        self.bases3Uri = args.modelpath if args.modelpath.endswith("/") else args.modelpath + "/"
        time.tzset()
        self.bases3Uri += "{}_{}/".format(time.strftime("%Y-%m-%d_%H-%M-%S"),os.uname().nodename)
        self.queue = Queue()
        self.threads = self.__postInit()
        self.logger = logging.getLogger(type(self).__name__)
        
    def __postInit(self) -> List[threading.Thread]:
        """
        PostInit mainly to initialize copy_to_s3 in thread

        Returns:
            List[threading.Thread]: list of Thread that run copy_to_s3 in each
        """
        retVal = []
        for i in range(1,4):
            thread = threading.Thread(target=self.copy_to_s3, args=(f"{type(self).__name__}-{i}",self.queue,))
            thread.daemon = True
            thread.start()
            print(f"Thread {type(self).__name__}-{i} started", Log.INF)
            retVal.append(thread)
        return retVal
        
        
    def on_any_event(self, event: FileSystemEvent) -> None:
        '''
        Event handler for watchdog 
        
        Parameters
        ---------
            event: https://pythonhosted.org/watchdog/api.html#watchdog.events.FileSystemEvent
        '''
        if not event.is_directory and (event.event_type == 'created' or event.event_type == 'modified'):
            self.queue.put(event.src_path)


    def copy_to_s3(self,tname,q):
        while True:
            if not q.empty():
                src_path = q.get()
                if src_path is None:
                    break
                dst_path = self.bases3Uri + src_path.replace("/saves/", "")
                logging.debug(f"{tname} is copying {src_path} to {dst_path}", Log.INF)
                self.s3.put(src_path, dst_path)
                q.task_done()
    
if args.modelpath.startswith("s3://"):
    logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')
    
    event_handler = SaveFileHandler()
    observer = Observer()
    watch = observer.schedule(event_handler, "/saves", recursive=True)
    observer.add_handler_for_watch(LoggingEventHandler(), watch)

def main(args: centralConfigLoader.CentralConfig):
    # accelerator
    fabric = L.Fabric(accelerator="gpu" if torch.cuda.is_available() else "cpu", devices=1, precision="32-true")
    fabric.launch()

    # webserver
    api_prefix = "/api/v1"
    app = FastAPI(title="split-learning", openapi_url=f"{api_prefix}/openapi.json")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    manager = ConnectionManager()
    os.makedirs("/saves/tb", exist_ok=True)
    os.makedirs("/saves/mlflow", exist_ok=True)
    os.makedirs("/saves/centralSaves", exist_ok=True)
    os.makedirs("/saves/inputModels", exist_ok=True)
    os.makedirs("/saves/outputModels", exist_ok=True)
    tbWriter = SummaryWriter(log_dir="/saves/tb")
    mlflow.set_tracking_uri(
        "file:///saves/mlflow"
    )  # Use the MLflow service name defined in Docker Compose
    mlflow.set_experiment("split-learning-experiment")

    if args.modelpath.startswith("s3://"):
        observer.start()


    dBase = db.Backend(username=args.userName, password=args.password)
    while True:
        if dBase.connect() == 0:
            dBase.flushKeys()
            break
        else:
            print("Connection Failed, retrying...", Log.ERR)
            time.sleep(10)

    @app.websocket("/ws/{name}", api_prefix)
    async def websocket_endpoint(websocket: WebSocket, name: str):
        await manager.connect(websocket)
        # ! Load model and optimizer.
        model = MUFASA()
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
        model, optimizer = fabric.setup(model, optimizer)
        itr = 0
        if os.path.exists(f"/saves/centralSaves/{name}.save.pth"):
            state = fabric.load(f"/saves/centralSaves/{name}.save.pth")
            model.load_state_dict(state["model"])
            optimizer.load_state_dict(state["optim"])
            itr = state["itr"]

        try:
            while True:
                optimizer.zero_grad()
                # * Receive logits1 from HIE
                messages_bytes = await websocket.receive_bytes()
                message = decode_message_b64(decompress(messages_bytes))
                if message.type == MessageType.TRAIN:
                    logits1 = deserialize_tensor(
                        message.raw["logits1"], dtype=torch.float32
                    )
                    logits1 = logits1.to(fabric.device)
                    logits1 = logits1.reshape(*message.data["tensor_shape"])

                    # * BulkModel forward
                    model.train()
                    logits1.requires_grad = True
                    logits2 = model(*logits1)

                    # * Send logits2 to HIE
                    logits2Data = logits2.detach().clone()
                    logits2Ser = serialize_tensor(logits2Data.cpu())
                    logits2Message = WSMessage(
                        type=MessageType.LOGITS,
                        data={"tensor_shape": logits2Data.shape},
                        raw={"logits2": logits2Ser},
                    )
                    logits2SendReq = encode_message_b64(logits2Message)
                    await websocket.send_bytes(compress(logits2SendReq))
                    # ? Flow to HIE
                    # * Receive logits2 grads
                    response_bytes = await websocket.receive_bytes()
                    response = decode_message_b64(decompress(response_bytes))

                    logits2Grad = deserialize_tensor(
                        response.raw["logits2Grad"], dtype=torch.float32
                    )
                    logits2Grad = logits2Grad.reshape(
                        *response.data["logits2GradShape"]
                    )
                    logits2Grad = logits2Grad.to(fabric.device)
                    fabric.backward(logits2, logits2Grad)
                    optimizer.step()

                    # * Sending logits1 grad back
                    logits1Grad = logits1.grad.detach().clone()
                    logits1GradSer = serialize_tensor(logits1Grad.cpu())
                    logits1GradMessage = WSMessage(
                        type=MessageType.GRADS,
                        data={"logits1GradShape": logits1.grad.shape},
                        raw={"logits1Grad": logits1GradSer},
                    )
                    logits1GradSendReq = encode_message_b64(logits1GradMessage)
                    await websocket.send_bytes(compress(logits1GradSendReq))
                    # ? Flow to HIE
                elif message.type == MessageType.VAL:
                    # * Get Logits1 for Validation
                    logits1 = deserialize_tensor(
                        message.raw["logits1"], dtype=torch.float32
                    )
                    logits1 = logits1.to(fabric.device)
                    logits1 = logits1.reshape(*message.data["tensor_shape"])

                    # * BulkModel forward
                    model.eval()
                    logits1.requires_grad = False
                    logits2 = model(*logits1)

                    # * Send logits2 to HIE
                    logits2Data = logits2.detach().clone()
                    logits2Ser = serialize_tensor(logits2Data.cpu())
                    logits2Message = WSMessage(
                        type=MessageType.LOGITS,
                        data={"tensor_shape": logits2Data.shape},
                        raw={"logits2": logits2Ser},
                    )
                    logits2SendReq = encode_message_b64(logits2Message)
                    await websocket.send_bytes(compress(logits2SendReq))
                    # ? Send logits2 back to client
        except WebSocketDisconnect:
            manager.disconnect(websocket)
            itr += 1
            # ! Save model, opt and epoch #.
            fabric.save(
                path=f"/saves/centralSaves/{name}.save.pth",
                state={"itr": itr, "model": model, "optim": optimizer},
            )
            print("Client Disconnected", Log.WRN)
        except Exception as e:
            print(str(e), Log.ERR)
            raise e

    @app.get("/status/{name}")
    def status(name: str):
        """
        Returns the position status to HIEs.
        If current Node is None and next node is a specific HIE node, that node has to start the training.
        """
        if not dBase.checkIfMember(name):
            raise HTTPException(
                status_code=401, detail="Your node not in registered list."
            )
        cur = dBase.getCurrentNode()
        nex = dBase.getNextNode()
        if cur == name and nex is None:
            raise HTTPException(status_code=202, detail="You are being trained")
        elif cur is None and nex == name:
            raise HTTPException(status_code=200, detail="You can start training")
        else:
            raise HTTPException(status_code=425, detail="Not your turn yet")

    @app.get("/register/{name}/{cont}-{categ}-{clin}-{out}")
    def connect(name: str, cont: int, categ: int, clin: int, out: int):
        """
        Saves the received name as nodeID in the node list.
        That nodeID is used by that node to identify itself uniquely.
        If duplicate nodeID is received, it is assumed that a reconnection is attempted.
        After enough reg requests are received, the registration is closed, and the status vars are set.
        If a new nodeID is received after the registration is closed, it is denied.
        """
        if not dBase.registrationOpen:
            if name.encode() in dBase.getNodeList():
                pass
            else:
                print(
                    f"Register requested by {name} after registration is closed.",
                    Log.ERR,
                )
                raise HTTPException(status_code=410, detail="Registration Closed")
        ret = dBase.addNodeToList(name)
        if ret == 1:
            print(f"Added {name} to Node List", Log.SUC)
            print([cont, categ, clin, out])
            dBase.storeModelIOSize(name, cont, categ, clin, out)
            if dBase.getNodeCount() == (args.world_size - 1):
                print(
                    f"All {dBase.getNodeCount()} nodes connected to Central.",
                    Log.SUC,
                )
                dBase.registrationOpen = False
                print("Starting train process soon...", Log.SUC)
                dBase.setNextNode(dBase.iterNextNodeID()[0])
            return 1
        elif ret == 0:
            print(f"Node {name} already in node list", Log.WRN)
            raise HTTPException(
                status_code=409, detail="Already registered with this ID"
            )
        elif ret not in [0, 1]:
            print(f"Unreacheable reached while adding {name} to node list", Log.ERR)
            print(f"{ret} returned from database", Log.WRN)
            raise HTTPException(status_code=501, detail="Unreacheable reached!!")

    @app.get("/lock/{name}")
    def lockForTraining(name: str):
        """
        This locks the Central server to respond to any other train requests.
        Analogous to global lock in python.
        """
        if not dBase.checkIfMember(name):
            raise HTTPException(
                status_code=401, detail="Your node not in registered list."
            )
        if dBase.getNextNode() == name and dBase.getCurrentNode() is None:
            print(f"{name} locked central for training.", Log.WRN)
            dBase.clearNextNode()
            dBase.setCurrentNode(name)
            return 1
        elif (dBase.getNextNode() is None) and dBase.getCurrentNode() == name:
            print(
                f"{name} was training and got disconnected, reconnected now...", Log.WRN
            )
            raise HTTPException(status_code=202, detail="Continue to train")
        else:
            print(f"Illegal LOCK requested from {name}", Log.ERR)
            raise HTTPException(status_code=403, detail="Lock requested too early")

    @app.get("/unlock/{name}")
    def unlock(name: str):
        """
        Unlocks Central server, which also triggers to update status.
        """
        if not dBase.checkIfMember(name):
            raise HTTPException(
                status_code=401, detail="Your node not in registered list."
            )
        if dBase.getCurrentNode() == name and dBase.getNextNode() is None:
            print(f"Unlock requested from {name}", Log.WRN)
            dBase.clearCurrentNode()
            nextNode, loc = dBase.iterNextNodeID()  #! TODO Fix
            dBase.setNextNode(nextNode)
            if loc == 0:
                print(f"{name} is last to train in this iter.")
        else:
            print(f"Illegal UNLOCK requested from {name}", Log.ERR)
            raise HTTPException(status_code=403, detail="Central not locked by self")

    @app.get("/erase/{name}")
    def eraseNodeID(name: str):
        """
        Removes the received nodeID from registrant list.
        """
        if not dBase.checkIfMember(name):
            raise HTTPException(
                status_code=401, detail="Your node not in registered list."
            )
        if dBase.removeNodeID(name) == 1:
            print(f"Removed {name}", Log.SUC)
        else:
            print(f"{name} asked for removal but not found", Log.WRN)
            raise HTTPException(status_code=409, detail="NodeID not in reg list")
        if dBase.getNodeCount() == 0:
            print("All nodes left, quitting program.", Log.WRN)
            print("Fusing models...", Log.WRN)
            # ? Combining central models for each HIE
            #added map_location to solve contradiction in model loading if 1 is trained on gpu and another loading on cpu
            map_location = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            saves = os.listdir("/saves/centralSaves/")
            stateDicts = []
            for save in saves:
                #state = fabric.load(f"/saves/centralSaves/{save}")
                state = torch.load(f"/saves/centralSaves/{save}", map_location=map_location)
                model = state["model"]
                stateDicts.append(model)
            keys = model.keys()
            del model
            modelState = MUFASA().state_dict()
            for key in keys:
                modelState[key] = torch.mean(
                    torch.stack([a[key] for a in stateDicts]), dim=0
                )
            torch.save(modelState, "/saves/finalCentral.pth")

            # ? Combining input and output models of all HIEs.

            inputModels = os.listdir("/saves/inputModels/")
            outputModels = os.listdir("/saves/outputModels/")

            inputStateDicts = []
            outputStateDicts = []
            for save in inputModels:
                with gzip.open("/saves/inputModels/" + save) as m:
                    inputStateDicts.append(torch.load(m, map_location=map_location))

            for save in outputModels:
                with gzip.open("/saves/outputModels/" + save) as m:
                    outputStateDicts.append(torch.load(m, map_location=map_location))
            cont, categ, clin, out, *tx = dBase.getModelIOSize(name)
            # print([cont, categ, clin, out])
            sampleInput = InputEncoder(int(cont), int(categ), int(clin)).state_dict()
            sampleOutput = OutputDecoder(
                int(out)
            ).state_dict()  # bring output shape from client outputModel = OutputDecoder(len(label_encoder.classes_))

            for key in sampleInput.keys():
                sampleInput[key] = torch.mean(
                    torch.stack([a[key] for a in inputStateDicts]), dim=0
                )

            for key in sampleOutput.keys():
                sampleOutput[key] = torch.mean(
                    torch.stack([a[key] for a in outputStateDicts]), dim=0
                )

            torch.save(sampleInput, "/saves/finalInput.pth")
            torch.save(sampleOutput, "/saves/finalOutput.pth")
            #print("Fusing Done", Log.SUC)
            bulkDict = torch.load("/saves/finalCentral.pth")
            inputDict = torch.load("/saves/finalInput.pth")
            outputDict = torch.load("/saves/finalOutput.pth")

            bulkDict.update(inputDict)
            bulkDict.update(outputDict)

            testModel = MUFASAFull(int(cont), int(categ), int(clin), int(out))
            testModel.load_state_dict(bulkDict)

            torch.save(bulkDict, "/saves/ONCFinal.pth")
            print("Fusing Done", Log.SUC)
                    
            if args.modelpath.startswith("s3://"):
                observer.stop()
                observer.join()
                [event_handler.queue.put(None) for i in range(len(event_handler.threads))]
                [t.join() for t in event_handler.threads]
            [print(f"File: {file}, Size: {os.path.getsize(os.path.join('/saves/', file))} bytes") for file in os.listdir('/saves/') if os.path.isfile(os.path.join('/saves/', file))]
            print("Quitting Program", Log.WRN)
            dBase.flushKeys()
            dBase.close()
            tbWriter.close()
            os.kill(os.getpid(), signal.SIGINT)
            # sys.exit(0)
            end_time = time.time()
            duration = end_time - start_time
            print("Program took", duration, "seconds to complete.")

    @app.put("/sendWeights/{name}/{part}")
    def receiveWeights(name: str, part: str, item: UploadFile):
        """
        Receives weights from the stream and saves it to a local file.
        """
        if not dBase.checkIfMember(name):
            raise HTTPException(
                status_code=401, detail="Your node not in registered list."
            )
        if dBase.getCurrentNode() != name and False:
            print(f"Illegal weights received from {name}", Log.ERR)
            raise HTTPException(status_code=403, detail="Not your turn yet")
        else:
            if part == "inp":
                savePath = f"/saves/inputModels/{name}_hieINP.pth"
            elif part == "out":
                savePath = f"/saves/outputModels/{name}_hieOUT.pth"
            else:
                raise HTTPException(
                    status_code=400, detail="Wrong weight type reeieved in request."
                )
            with open(savePath, "wb") as f:
                f.write(item.file.read())

    @app.get("/getWeights/{name}/{part}")
    def giveWeigts(name: str, part: str):
        """
        Sends saved weights to requested node.
        """
        if not dBase.checkIfMember(name):
            raise HTTPException(
                status_code=401, detail="Your node not in registered list."
            )
        if dBase.getCurrentNode() != name:
            print(f"Received illegal weights request from {name}")
            return HTTPException(status_code=403, detail="Illegaly requested weights")
        if part == "inp":
            savePath = "hieINP.pth"
        elif part == "out":
            savePath = "hieOUT.pth"
        else:
            raise HTTPException(
                status_code=400, detail="Wrong weight type receieved in request."
            )
        if not os.path.exists(savePath):
            raise HTTPException(
                status_code=204,
                detail="No weights found in central, you may be the first one to train",
            )
        return FileResponse(savePath, media_type="application/octet-stream")
#############################################################################################
#  please add s3 handler to the below to save plots to s3 saves

    @app.post("/sendroc/{name}")
    async def roc(name: str, req: Request):
        data = await req.body()
        data = json.loads(decompress(data))
        tpr = data["tpr"]
        fpr = data["fpr"]
        cm_list = data["cm"]
        #cm = np.array(data["cm"])
        # print(data["rec"])
        # print(data['pre'])
        rec = data["rec"]
        pre = data["pre"]
        plt.plot(fpr, tpr, label="ROC Curve" )
        plt.plot([0, 1], [0, 1], 'k--', label='No Skill')
        plt.xlim([-0.1, 1.05])
        plt.ylim([-0.1, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve for {name}')
        plt.legend()
        plt.savefig(f"/saves/roc_{name}.png")
        plt.close()
        plt.clf()
        plt.plot(rec, pre, label="PR Curve" )
        # plt.plot([0, 1], [0, 1], 'k--', label='No Skill')
        plt.xlim([-0.1, 1.05])
        plt.ylim([-0.1, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'PR Curve for {name}')
        plt.legend()
        plt.savefig(f"/saves/pr_{name}.png")
        plt.close()
        plt.clf()

        f, axes = plt.subplots(1, 5, figsize=(25, 15))
        axes = axes.ravel()
        # Plot each confusion matrix
        for i, cm in enumerate(cm_list):
            cm_array = np.array(cm)  # Convert to numpy array
            display_labels = np.arange(cm_array.shape[0])  # Create display labels based on the number of classes
            disp = ConfusionMatrixDisplay(confusion_matrix=cm_array, display_labels=display_labels)
            disp.plot(ax=axes[i])
            axes[i].set_title(f'Label {i+1}')

        # Add label to the confusion matrix plot
        plt.suptitle('Confusion Matrices', ha='center', va='center')
       
        plt.savefig(f"/saves/confusion_matrix_{name}.png")
        plt.close()
        plt.clf()

    @app.post("/sendMetrics/{name}")
    async def receiveMetrics(name: str, req: Request):
        if not dBase.checkIfMember(name):
            raise HTTPException(
                status_code=401, detail="Your node not in registered list."
            )
        if dBase.getCurrentNode() != name:
            print(f"Received illegal metrics from {name}")
            return HTTPException(status_code=403, detail="Illegaly requested weights")
        data = await req.body()
        data = json.loads(decompress(data))
        curTestAcc = data["teAcc"][-1]
        # MLflow Tracking Integration
        with mlflow.start_run(
            run_name=name
        ):  # Run names should be unique within an experiment
            for i in range(len(data["trAcc"])):
                mlflow.log_metrics(
                    {
                        "train_loss": data["trLoss"][i],
                        "train_accuracy": data["trAcc"][i],
                        "test_accuracy": data["teAcc"][i],
                        #"privacy_score1": data["privacyScore1"][i],
                        #"privacy_score2": data["privacyScore2"][i],
                        "precision_scores": data["precisionScores"][i],
                        "recall_scores": data["recallScores"][i],
                        "f1_scores": data["f1Scores"][i],
                    },
                    step=i,
                )

                tbWriter.add_scalars(
                    "combined",
                    {
                        "Train Loss": data["trLoss"][i],
                        "Train Acc": data["trAcc"][i],
                        "Test Acc": data["teAcc"][i],
                    },
                    dBase.iterTBVariable(),
                )
                tbWriter.add_scalars(
                    name,
                    {
                        "Train Loss": data["trLoss"][i],
                        "Train Acc": data["trAcc"][i],
                        "Test Acc": data["teAcc"][i],
                    },
                    dBase.iterTBVariable(name),
                )
            if curTestAcc > dBase.getBestTestAcc():
                dBase.setBestTestAcc(curTestAcc)
                mlflow.log_metric("best_test_accuracy", curTestAcc)
                # shutil.copyfile("./hieINP.pth", "/saves/hieINP.best.pth")
                # shutil.copyfile("./hieOUT.pth", "/saves/hieOut.best.pth")
                # torch.save(model.state_dict(), "/saves/cent.best.pth")
                raise HTTPException(status_code=201, detail="New best model")

    def compress(data: Any):
        return gzip.compress(data)

    def decompress(data: Any):
        return gzip.decompress(data)

    # @app.websocket("/con")
    # def connection():
    #     ...

    addr: str = args.address
    port: int = args.port
    server_config = Config(
        app=app,
        host=addr,
        port=port,
        ws_max_size=64 * 1024 * 1024,
        ws_ping_interval=20,
        ws_ping_timeout=60,
    )  # fixed ping timeout
    server = Server(config=server_config)
    server.run()


if __name__ == "__main__":
    main(args)