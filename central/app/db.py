import redis


class Backend:
    """
    Database backend for Central Server. Manages reading from and writing to Redis Database.

    Uses Redis' Set data structure to store the registrants name.
    So, duplicate entried are cleared automatically.
    Also uses the string datatype to store current and next nodeID entries.

    Eventhough these methods are called asynchronously and multiple times through central,
    they don't run into race condition as each call to redis is handled atomically on redis side.

    args:
    prefix: str = an additional string added to a key at the beginning. A standard Redis Practice.
        Helps distinguish keys belonging other programs and allows for wildcard matching.
        Not necessary in the current case, but implemented to standard.
    """

    def __init__(self, username: str, password: str, prefix: str = "onc"):
        self.prefix = prefix
        self.connection = (
            None  # Redis connection object, inited when self.connect is called
        )
        self.sscanCursor = 0  # Node list iterator index
        self.kNodeList = "nodeList"  # Key string for storing the node list
        self.kCurrentNode = (
            "currentNode"  # Key string for storing currently trained node
        )
        self.kNextNode = "nextNode"  # Key string for storing next node in line
        self.kBestAcc = "bestTestAcc"
        self.kTBWriterIndex = "combined"
        self.kModelIOSize = "modelSizes"
        self.nodes = None  # Nodes list, assigned the list after registration is closed
        self.registrationOpen = True  # Used for handle registrations
        self.psd = password
        self.usr = username

    def connect(self):
        """
        Opens a connection to the local Redis Database and tests if it is online.
        """
        try:
            self.connection = redis.Redis(
                host="localhost", port=6499, password=self.psd + " ", username=self.usr
            )  # TODO ADD Password
            self.connection.set(self.getKey("a"), "test")
            self.connection.delete(self.getKey("a"))
        except redis.exceptions.ConnectionError as e:
            # print("Unable to connect to Database, retry connection.")
            print(e)
            self.connection.close()
            return 1
        return 0

    def getKey(self, key: str, prefix2: str = ""):
        """
        Returns the key in Redis convention.
        If the key is "kNextNode", final key will be "onc:kNextNode"
        """
        return self.prefix + ":" + (f"{prefix2}:" if prefix2 else "") + key

    def getNodeCount(self):
        """
        Returns the number of nodes currently registered. Uses `scard` function (Set CARDinality).
        """
        return self.connection.scard(self.getKey(self.kNodeList))
        # return self.connection.llen(self.getKey(self.kNodeList))

    def getCurrentNode(self):
        """
        Returns the current node being trained. Uses the `get` function.
        """
        nodeID = self.connection.get(self.getKey(self.kCurrentNode))
        if nodeID:
            return nodeID.decode()
        else:
            return nodeID

    def getNextNode(self):
        """
        Returns the next node to train. Uses the `get` function
        """
        nodeID = self.connection.get(self.getKey(self.kNextNode))
        if nodeID:
            return nodeID.decode()
        else:
            return nodeID

    def getNodeList(self):
        """
        Returns the list of nodes registered. Uses the `smembers` function (Set MEMBERS).
        """
        return self.connection.smembers(self.getKey(self.kNodeList))

    def addNodeToList(self, nodeID: str):
        """
        Adds given node ID to the registration set/list. Uses the `sadd` function (Set ADD).
        """
        return self.connection.sadd(self.getKey(self.kNodeList), nodeID)

    def flushKeys(self):
        """
        Removes all the keys created by this program.
        `unlink` is an async version of `delete` function.
        """
        return self.connection.unlink(
            self.getKey(self.kNodeList),
            self.getKey(self.kCurrentNode),
            self.getKey(self.kNextNode),
        )

    def close(self):
        """
        Closes connection to the local Redis Database.
        Removes all the keys and drop the connection.
        """
        self.flushKeys()
        self.connection.close()

    def getIterOverMems(self):
        """
        Returns an iterator over the list of nodeIDs.
        Uses the `sscan_iter` function (Set SCAN Iterator).
        """
        return self.connection.sscan_iter(self.getKey(self.kNodeList), count=1)

    def setCurrentNode(self, nodeID: str):
        """
        Sets currently trained nodeID with given ID. Uses `set` function.
        """
        return self.connection.set(self.getKey(self.kCurrentNode), nodeID)

    def setNextNode(self, nodeID: str):
        """
        Sets next node to train with given ID. Uses `set` function.
        """
        return self.connection.set(self.getKey(self.kNextNode), nodeID)

    def clearCurrentNode(self):
        """
        Removes currently trained nodeID. Uses `delete` function.
        """
        return self.connection.delete(self.getKey(self.kCurrentNode))

    def clearNextNode(self):
        """
        Removed next nodeID to train. Uses `delete` function.
        """
        return self.connection.delete(self.getKey(self.kNextNode))

    def iterNextNodeID(self):
        """
        Returns one nodeID everytime this method is called, circularly.
        Index goes back to 0 when limit is hit.
        This is how round robin is operated while maintaning consistant order.
        """
        if not self.nodes:
            self.nodes = list(self.connection.smembers(self.getKey(self.kNodeList)))
            self.sscanCursor = 0
        else:
            self.sscanCursor = (self.sscanCursor + 1) % len(self.nodes)
        return self.nodes[self.sscanCursor].decode(), self.sscanCursor

    def removeNodeID(self, nodeID: str):
        """
        Removed given nodeID from registration list. Uses `srem` function (Set REMove).
        """
        return self.connection.srem(self.getKey(self.kNodeList), nodeID)

    def checkIfMember(self, nodeID: str):
        """
        Checks if nodeID is present in the registrant list.
        Uses `sismember` function (Set IS MEMBER)
        """
        return (
            True
            if self.connection.sismember(self.getKey(self.kNodeList), nodeID) == 1
            else 0
        )

    def setBestTestAcc(self, acc: float):
        """
        Sets the best test accuracy with given value. Uses `set` function.
        """
        return self.connection.set(self.getKey(self.kBestAcc), acc)

    def getBestTestAcc(self):
        """
        Returns the best test accuracy. Uses `get` function.
        """
        x = self.connection.get(self.getKey(self.kBestAcc))
        if x:
            return float(x.decode())
        return 0

    def iterTBVariable(self, name: str = ""):
        """
        Keeps track, iterates and returns the index for logging to tensorboard.
        Uses `incr`(INCRease) function to iterate up and `get` to get and return the index value.
        """
        if name:
            self.connection.incr(self.getKey(name, prefix2="TB"))
            return int(self.connection.get(self.getKey(name, prefix2="TB")).decode())
        else:
            self.connection.incr(self.getKey(self.kTBWriterIndex, prefix2="TB"))
            return int(
                self.connection.get(
                    self.getKey(self.kTBWriterIndex, prefix2="TB")
                ).decode()
            )

    def storeModelIOSize(self, name, cont, categ, clin, out):
        return self.connection.rpush(
            self.getKey(name, self.kModelIOSize), cont, categ, clin, out
        )

    def getModelIOSize(self, name):
        return list(self.connection.lrange(self.getKey(name, self.kModelIOSize), 0, -1))
