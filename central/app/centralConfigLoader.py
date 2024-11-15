import importlib
import os
from debugPrint import genDebugFunction, Log

print = genDebugFunction(True, True, saveToFile="central.out")


class CentralConfig:
    def __init__(self, configFileLocation: str = None):
        if configFileLocation:
            configFileLocation = str(configFileLocation)
            self.modelpath = os.getenv("MODEL_PATH", "saves")
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
                self.userName = conf.userName
                self.password = conf.userPass
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
                self.datapath = (
                    int(os.environ["DATA_PATH"])
                    if "DATA_PATH" in os.environ
                    else "data/"
                )
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
                if "UUSR" in os.environ:
                    self.userName = os.environ["UUSR"]
                else:
                    raise Exception("Username not found in env vars.")

                if "UPSD" in os.environ:
                    self.password = os.environ["UPSD"]
                else:
                    raise Exception("Password not found in env vars.")
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
