import asyncio
import gzip
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict

import dataLoader
import hieConfigLoader
import lightning as L
import pandas as pd
import requests
import torch
import websockets
from debugPrint import Log, genDebugFunction
from pydantic import BaseModel
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm.auto import tqdm

print = genDebugFunction(printTime=True, printDelta=True)
import lightning as L
import split_learning
from debugPrint import Log, genDebugFunction
from pydantic import BaseModel
import torch.nn.functional as F

# from split_learning.models.bioClient import InputEncoder, OutputDecoder
from split_learning.models.mufasaHIE import InputEncoder, OutputDecoder
from split_learning.schemas.message import MessageType, WSMessage
from split_learning.utils.serde import (
    decode_message_b64,
    deserialize_tensor,
    encode_message_b64,
    serialize_tensor,
)

from noPeekLoss import NoPeekTripleLoss
#from scores import klDiv, rocVals
import matplotlib.pyplot as plt

print = genDebugFunction(printTime=True, printDelta=True)

from sklearn.metrics import recall_score, precision_score, f1_score, roc_curve, precision_recall_curve,confusion_matrix
import numpy as np

null = None
if "CONFIG" in os.environ:
    args = hieConfigLoader.HIEConfig(str(os.environ["CONFIG"]))
else:
    args = hieConfigLoader.HIEConfig("confHIE.py")


class StatePacket(BaseModel):
    state: Dict[Any, Any] = None


def main(args: hieConfigLoader.HIEConfig):
    # accelerator
    fabric = L.Fabric(accelerator="gpu" if torch.cuda.is_available() else "cpu", devices=1, precision="32-true")
    fabric.launch()

    print("Loading data")
    # trainLoader, testLoader = dataLoader.loadBioData()
    # trainLoader, testLoader = dataLoader.loadRandomDataset()
    trainLoader, testLoader, labelEncoders, shapes = dataLoader.loadHIEDATA(args.datapath)
    print("Done loading data", Log.SUC)
    trainLoader, testLoader = fabric.setup_dataloaders(trainLoader, testLoader)

    inputModel = InputEncoder(*shapes[:3])
    outputModel = OutputDecoder(shapes[3])
    # outputModel = OutputDecoder(len(label_encoder.classes_))#AttributeError: 'LabelEncoder' object has no attribute 'classes_' in dataloader.py
    inputOPT = torch.optim.SGD(inputModel.parameters(), lr=args.lr, momentum=0.9)
    outputOPT = torch.optim.SGD(outputModel.parameters(), lr=args.lr, momentum=0.9)
    inputModel, inputOPT = fabric.setup(inputModel, inputOPT)
    outputModel, outputOPT = fabric.setup(outputModel, outputOPT)
    # lossFN = torch.nn.CrossEntropyLoss()
    lossFN = NoPeekTripleLoss(dcor_weighting=0.0)
    addr: str = args.address
    port: int = args.port
    uri = f"ws://{addr}:{port}/ws/"

    def compress(data: Any):
        return gzip.compress(data)

    def decompress(data: Any):
        return gzip.decompress(data)

    def waitForTurn():
        while True:
            for _ in range(10):
                try:
                    resp = requests.get(
                        f"http://{addr}:{port}/status/{args.nodeID}",
                        timeout=60,  # changed from 20 to 60
                    )
                except requests.exceptions.Timeout as _:
                    print("Unable to connect to Central, timed-out", Log.ERR)
                    continue
                if resp.status_code in [200, 425, 202]:
                    break
                else:
                    print("Unable to connect to Central", Log.ERR)
                    time.sleep(20)
            else:
                print("Cannot connect to Central", Log.ERR)
                sys.exit(1)
            if resp.status_code == 200:
                print("Starting training for this iteration", Log.WRN)
                break
            elif resp.status_code == 425:
                print("Waiting for turn")
                time.sleep(30)
            elif resp.status_code == 202:
                print(
                    "This node already started training and has lock on Central",
                    Log.WRN,
                )
                break

    def lockCentral():
        for _ in range(10):
            try:
                resp = requests.get(
                    f"http://{addr}:{port}/lock/{args.nodeID}", timeout=10
                )
            except requests.exceptions.Timeout as _:
                print("Unable to clock central, timed-out", Log.ERR)
            if resp.status_code in [200, 202]:
                break
            elif resp.status_code == 403:
                print("Lock requested too early, waiting for chance")
                waitForTurn()
            else:
                print("unable to connect to Central")
                time.sleep(20)
        else:
            print("Cannot lock to Central", Log.ERR)
            sys.exit(1)

    def getWeights(part: str):
        for _ in range(10):
            try:
                weightResponse = requests.get(
                    f"http://{addr}:{port}/getWeights/{args.nodeID}/{part[:3]}",
                    timeout=120,
                )
                break
            except requests.exceptions.Timeout as _:
                print("Unable to get weights, timed-out", Log.ERR)
                continue

        if weightResponse.status_code == 204:
            print("This is the first client to train.", Log.WRN)
        else:
            with open("hieSave.torch", "wb") as f:
                f.write(weightResponse.content)
            print("Loaded trained weights from central", Log.SUC)
            with gzip.open("hieSave.torch", "rb") as g:
                return torch.load(g)

    def sendWeights(model: torch.nn.Module, part: str):
        with gzip.open("model.torch", "wb") as g:
            torch.save(model.state_dict(), g, pickle_protocol=5)
        for _ in range(10):
            try:
                resp = requests.put(
                    f"http://{addr}:{port}/sendWeights/{args.nodeID}/{part[:3]}",
                    files={"item": open("model.torch", "rb")},
                    timeout=120,
                )
            except requests.exceptions.Timeout as e:
                print("Took too long to upoad wights.")
                print(e)
                time.sleep(30)
            if resp.status_code == 403:
                print("Not our turn yet, discarding weights", Log.ERR)
                break
            elif resp.status_code == 200:
                print("Uploaded weights to central server.", Log.SUC)
                break
        else:
            print("Unable to uploaded weights", Log.ERR)

    def unlockCentral():
        print("Unlocking Central")
        for _ in range(10):
            try:
                resp = requests.get(
                    f"http://{addr}:{port}/unlock/{args.nodeID}", timeout=10
                )
                if resp.status_code == 200:
                    break
                elif resp.status_code == 403:
                    print("Not this node's turn to unlock, skipping", Log.ERR)
            except requests.exceptions.Timeout as _:
                print("Unable to connect to central", Log.ERR)
                time.sleep(10)
                continue

    def sendMetrics(
        trainLoss,
        trainAcc,
        testAcc,
        #privScores1,
        #privScores2,
        f1Scores,
        precisionScores,
        recallScores,
    ):
        for _ in range(10):
            try:
                requests.post(
                    f"http://{addr}:{port}/sendMetrics/{args.nodeID}",
                    timeout=20,
                    data=compress(
                        json.dumps(
                            {
                                "trLoss": trainLoss,
                                "trAcc": trainAcc,
                                "teAcc": testAcc,
                                #"privacyScore1": privScores1,
                                #"privacyScore2": privScores2,
                                "recallScores": recallScores,
                                "precisionScores": precisionScores,
                                "f1Scores": f1Scores,
                            }
                        ).encode("utf-8")
                    ),
                )
                if resp.status_code == 200:
                    break
                elif resp.status_code == 201:
                    print("New Best Model Saved")
                    break
            except requests.exceptions.Timeout as _:
                print("Unable to connect to central", Log.ERR)
                time.sleep(10)
                continue

    # def sendMetrics(trainLoss, trainAcc, testAcc):
    #     for _ in range(10):
    #         try:
    #             requests.post(
    #                 f"http://{addr}:{port}/sendMetrics/{args.nodeID}",
    #                 timeout=20,
    #                 data=compress(
    #                     json.dumps(
    #                         {"trLoss": trainLoss, "trAcc": trainAcc, "teAcc": testAcc}
    #                     ).encode("utf-8")
    #                 ),
    #             )
    #             if resp.status_code == 200:
    #                 break
    #             elif resp.status_code == 201:
    #                 print("New Best Model Saved")
    #                 break
    #         except requests.exceptions.Timeout as e:
    #             print("Unable to connect to central", Log.ERR)
    #             time.sleep(10)
    #             continue

    async def train_splitnn():
        for itr in range(args.iterations):
            print(f"Awaiting Iteration {itr} ...")
            waitForTurn()
            try:
                print(f"Connecting to {uri}")
                lockCentral()
                # ? disable loading weights, for privacy.
                # weights = getWeights("inp")
                # if weights:
                #     inputModel.load_state_dict(weights)
                # weights = getWeights("out")
                # if weights:
                #     outputModel.load_state_dict(weights)
                trainLoss = []
                trainAcc = []
                testAcc = []
                #privacyScores1 = []
                #privacyScores2 = []
                precisionScores = []
                recallScores = []
                f1Scores = []

                async with websockets.connect(
                    uri + args.nodeID, max_size=64 * 1024 * 1024
                ) as websocket:
                    for epoch in range(args.epochs):
                        first_ep_time = time.time()
                        running_loss = 0.0
                        runningAcc = 0
                        runningTestAcc = 0
                        runningTestPrecision = 0
                        runningTestRecall = 0
                        runningTestF1 = 0
                        #runningScore1 = 0.0
                        #runningScore2 = 0.0
                        totalTestSamples = 0
                        totalTrainSamples = 0
                        # pbar = tqdm(enumerate(trainLoader), total=len(trainLoader))
                        for _, data in enumerate(trainLoader):
                            cont, categ, clin, labels = data
                            labels = (
                                labels.long()
                            )  # Ensure labels are long type. added to solve a bug.
                            inputOPT.zero_grad()
                            outputOPT.zero_grad()

                            # * InputModel Forward
                            inputModel.train()
                            # print(len(images))
                            logits1 = inputModel(
                                [cont.float(), categ.float(), clin.float()]
                            )  # ! Change model to accept single tensor.

                            # * Send logits1 to Central
                            logits1Data = logits1.detach().clone()
                            logits1DataSer = serialize_tensor(logits1Data.cpu())
                            logits1Message = WSMessage(
                                type=MessageType.TRAIN,
                                data={"tensor_shape": logits1Data.shape},
                                raw={"logits1": logits1DataSer},
                            )
                            logits1SendReq = encode_message_b64(logits1Message)
                            await websocket.send(compress(logits1SendReq))
                            # ? Flow To Central
                            # * Receive logits2 from Central
                            response_byes = await websocket.recv()
                            response = decode_message_b64(decompress(response_byes))
                            logits2 = deserialize_tensor(response.raw["logits2"])
                            logits2 = logits2.to(fabric.device)
                            logits2 = logits2.reshape(*response.data["tensor_shape"])

                            # * OutputModel forward pass
                            outputModel.train()
                            logits2.requires_grad = True
                            logits3 = outputModel(logits2)

                            # * Loss calculation, backward and optimizer step.
                            # loss = lossFN(logits3, labels)
                            ceLoss, npLosses = lossFN(
                                [cont.float(), categ.float(), clin.float()],
                                logits1,
                                logits3,
                                labels,
                            )
                            if npLosses:
                                npLosses[0].backward(retain_graph=True)
                                npLosses[1].backward(retain_graph=True)
                                npLosses[2].backward(retain_graph=True)
                            fabric.backward(ceLoss)
                            outputOPT.step()

                            # * Send logits2 grads back to central
                            logits2Grads = logits2.grad.detach().clone()
                            serializedGrads = serialize_tensor(logits2Grads.cpu())
                            logits2GradMessage = WSMessage(
                                type=MessageType.GRADS,
                                data={"logits2GradShape": logits2.grad.shape},
                                raw={"logits2Grad": serializedGrads},
                            )
                            logits2GradSendReq = encode_message_b64(logits2GradMessage)
                            await websocket.send(compress(logits2GradSendReq))
                            # ? Flow to Central
                            # * Receive Logits1 grads
                            response = await websocket.recv()
                            response = decode_message_b64(decompress(response))
                            grads = deserialize_tensor(
                                response.raw["logits1Grad"], dtype=torch.float32
                            )
                            grads = grads.reshape(*response.data["logits1GradShape"])
                            grads = grads.to(fabric.device)
                            fabric.backward(logits1, grads)
                            inputOPT.step()

                            running_loss += ceLoss.item()
                            runningAcc += (
                                (
                                    (torch.nn.Sigmoid()(logits3)).round()
                                    == labels.unsqueeze(1)
                                )
                                .float()
                                .sum()
                            )
                            totalTrainSamples += torch.numel(logits3)
                        trainLoss.append(running_loss)
                        trainAcc.append(runningAcc.item() / (totalTrainSamples))
                        allPreds = []
                        allTrues=[]
                        for _, data in enumerate(testLoader):
                            cont, categ, clin, labels = data
                            inputOPT.zero_grad()
                            outputOPT.zero_grad()

                            # * InputModel Forward
                            inputModel.eval()
                            logits1 = inputModel(
                                [cont.float(), categ.float(), clin.float()]
                            )  #! Change model to accept single tensor.
                            # * Send logits1 to Central
                            logits1Data = logits1.detach().clone()
                            logits1DataSer = serialize_tensor(logits1Data.cpu())
                            logits1Message = WSMessage(
                                type=MessageType.VAL,
                                data={"tensor_shape": logits1Data.shape},
                                raw={"logits1": logits1DataSer},
                            )
                            logits1SendReq = encode_message_b64(logits1Message)
                            await websocket.send(compress(logits1SendReq))
                            # ? Flow To Central
                            # * Receive logits2 from Central
                            response_byes = await websocket.recv()
                            response = decode_message_b64(decompress(response_byes))
                            logits2 = deserialize_tensor(response.raw["logits2"])
                            logits2 = logits2.to(fabric.device)
                            logits2 = logits2.reshape(*response.data["tensor_shape"])
                            # * OutputModel forward pass
                            outputModel.eval()
                            logits2.requires_grad = False
                            logits3 = outputModel(logits2)
                            predLabels = F.sigmoid(logits3)
                            allTrues.extend(labels.flatten().cpu().tolist())
                            allPreds.extend(predLabels.flatten().cpu().tolist())
                            predLabels = predLabels.round()
                            runningTestAcc += (
                                ((torch.nn.Sigmoid()(logits3)).round() == labels)
                                .float()
                                .sum()
                            )
                            runningTestPrecision += precision_score(
                                labels.flatten().cpu().numpy(),
                                predLabels.detach().flatten().cpu().numpy(),
                                average="macro",
                                labels=torch.unique(labels).cpu().numpy(),
                                zero_division=np.nan,
                            )
                            runningTestRecall += recall_score(
                                labels.flatten().cpu().numpy(),
                                predLabels.detach().flatten().cpu().numpy(),
                                average="macro",
                                labels=torch.unique(labels).cpu().numpy(),
                                zero_division=np.nan,
                            )
                            #totalTestSamples = 0
                            runningTestF1 += f1_score(
                                labels.flatten().cpu().numpy(),
                                predLabels.detach().flatten().cpu().numpy(),
                                average="macro",
                                labels=torch.unique(labels).cpu().numpy(),
                                zero_division=np.nan,
                            )
                                                     
                            totalTestSamples += torch.numel(logits3)
                        # print(totalTestSamples)
                        testAcc.append(runningTestAcc.item() / totalTestSamples)
                        recallScores.append(runningTestRecall)
                        precisionScores.append(runningTestPrecision)
                        f1Scores.append(runningTestF1)
                        epoch_time = time.time() - first_ep_time
                        print(
                            f"Epoch: {epoch} TrLoss: {running_loss:.3f} TeAcc: {testAcc[-1]:.3f} time: {epoch_time:.3f}s recall: {recallScores[-1]:.3f} precision: {precisionScores[-1]:.3f} f1score: {f1Scores[-1]:.3f}"
                        )
                    print(f"Itr {itr} done")
                    # ? Disable weight sharing for privacy.
                    # sendWeights(inputModel, "inp")
                    # sendWeights(outputModel, "out")
                    sendMetrics(
                        trainLoss,
                        trainAcc,
                        testAcc,
                        #privacyScores1,
                        #privacyScores2,
                        f1Scores,
                        precisionScores,
                        recallScores,
                    )
                    unlockCentral()
            except ConnectionRefusedError:
                print("Connection Refused", Log.ERR)
            except Exception as e:
                print(str(e), Log.ERR)
                raise e
        ##? Un register from Central
        print("De-Registering from Central")
        sendWeights(inputModel, "inp")
        sendWeights(outputModel, "out")
        fpr, tpr, _ = roc_curve(allTrues, allPreds)
        precision, recall, _ = precision_recall_curve(allTrues, allPreds)

        # Reshape allTrues and allPreds to separate the labels
        num_labels = 5
        allTrues_reshaped = np.reshape(allTrues, (-1, num_labels))
        allPreds_reshaped = np.reshape(allPreds, (-1, num_labels))

        # Compute confusion matrices for each label
        confusion_matrices = []
        for i in range(num_labels):
            cm = confusion_matrix(allTrues_reshaped[:, i], np.round(allPreds_reshaped[:, i]))
            confusion_matrices.append(cm.tolist())

        #cm=confusion_matrix(allTrues, np.array(allPreds).round().tolist())
        #cm=cm.tolist()
        requests.post(f"http://{addr}:{port}/sendroc/{args.nodeID}", timeout=20, data=compress(json.dumps({"tpr": tpr.tolist(), "fpr":fpr.tolist(), "pre": precision.tolist(), "rec": recall.tolist(), "cm": confusion_matrices}).encode("utf-8")))
        requests.get(f"http://{addr}:{port}/erase/{args.nodeID}")
        print("Exiting Program", Log.SUC)

    for att in range(5):
        try:
            resp = requests.get(
                f"http://{addr}:{port}/register/{args.nodeID}/{shapes[0]}-{shapes[1]}-{shapes[2]}-{shapes[3]}",
                timeout=120,
            )
            if resp.status_code in [410, 409, 200, 501]:
                break
            else:
                print("Unknown response from Central, retrying...", Log.ERR)
                continue
        except (requests.ConnectTimeout, requests.exceptions.ConnectionError) as e:
            print(e, Log.ERR)
            print(f"Unable to connect to Central, attempt {att}, retrying...", Log.WRN)
            time.sleep(30)

    else:
        print("Unable to connect to Central, exiting program", Log.ERR)
        exit(1)
    if resp.status_code == 200:
        print("Registration successful", Log.SUC)
    elif resp.status_code == 409:
        print(
            f"{args.nodeID} already in registrants list, current node maybe reconnecting",
            Log.WRN,
        )
    elif resp.status_code == 410:
        print("Registration closed, contact admins", Log.ERR)
        sys.exit(1)
    elif resp.status_code == 501:
        print("Central not working properly, exitting program", Log.ERR)
        sys.exit(1)
    asyncio.get_event_loop().run_until_complete(train_splitnn())


if __name__ == "__main__":
    main(args=args)