import importlib
import os
from debugPrint import genDebugFunction, Log

print = genDebugFunction(True, True, saveToFile="central.out")


class HIEConfig:
    def __init__(self, configFileLocation: str = None):
        if configFileLocation:
            configFileLocation = str(configFileLocation)
            if configFileLocation.lower() != "os":
                configFileLocation = str(configFileLocation)
                print(f"Using {configFileLocation} for configuration")
                import sys

                if (dName := os.path.dirname(configFileLocation)) != "":
                    sys.path.append(dName)

                confSpec = importlib.util.find_spec(
                    os.path.basename(configFileLocation).split(".")[0]
                )

                conf = importlib.util.module_from_spec(confSpec)
                confSpec.loader.exec_module(conf)
                self.world_size = conf.world_size
                self.epochs = conf.epochs
                self.iterations = conf.iterations
                self.batch_size = conf.batch_size
                self.datapath = conf.datapath
                self.lr = conf.lr
                self.address = conf.address
                self.port = conf.port
                self.ifname = conf.ifname
                self.nodeID = conf.nodeID
                print("Config Loaded successfully", Log.SUC)
            elif configFileLocation.lower() == "os":
                print("Fetching config from environmental variables")
                self.world_size = (
                    int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 2
                )
                self.epochs = int(os.environ["EPOCHS"]) if "EPOCHS" in os.environ else 4
                self.iterations = (
                    int(os.environ["ITERATIONS"]) if "ITERATIONS" in os.environ else 3
                )
                self.batch_size = (
                    int(os.environ["BATCH_SIZE"]) if "BATCH_SIZE" in os.environ else 16
                )
                self.datapath = os.getenv("DATA_PATH", "./data")
                self.lr = float(os.environ["LR"]) if "LR" in os.environ else 0.001
                self.address = (
                    os.environ["CENTRAL_ADDRESS"]
                    if "CENTRAL_ADDRESS" in os.environ
                    else "localhost"
                )
                self.port = (
                    int(os.environ["CENTRAL_PORT"])
                    if "CENTRAL_PORT" in os.environ
                    else 7732
                )
                self.ifname = os.environ["IFNAME"] if "IFNAME" in os.environ else None
                if "NODEID" in os.environ:
                    self.nodeID = os.environ["NODEID"]
                elif "RANK" in os.environ:
                    self.nodeID = "hie" + str(os.environ["RANK"])
                else:
                    self.nodeID = "hie1"
            else:
                print("Fatal Error, this shouldn't be reached", Log.ERR)
                exit(1)
        else:
            print("Using default config", Log.WRN)
            self.world_size = 2
            self.epochs = 4
            self.iterations = 3
            self.batch_size = 16
            self.datapath = "data/"
            self.lr = 0.001
            self.address = "localhost"
            self.port = 7732
            self.ifname = "eth0"
            self.nodeID = "hie1"
