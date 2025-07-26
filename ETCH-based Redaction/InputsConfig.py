import random

class InputsConfig:

    """ Select the model to be simulated.
    0 : The base model
    1 : Bitcoin model
    2 : Ethereum model
    """
    model = 1

    ''' Input configurations for the base model '''
    # if model == 0:
    #
    #     ''' Block Parameters '''
    #     Binterval = 600  # Average time (in seconds)for creating a block in the blockchain
    #     Bsize = 1.0  # The block size in MB
    #     Bdelay = 0.42  # average block propogation delay in seconds, #Ref: https://bitslog.wordpress.com/2016/04/28/uncle-mining-an-ethereum-consensus-protocol-flaw/
    #     Breward = 12.5  # Reward for mining a block
    #
    #     ''' Transaction Parameters '''
    #     hasTrans = True  # True/False to enable/disable transactions in the simulator
    #     Ttechnique = "Light"  # Full/Light to specify the way of modelling transactions
    #     Tn = 10  # The rate of the number of transactions to be created per second
    #     # The average transaction propagation delay in seconds (Only if Full technique is used)
    #     Tdelay = 5.1
    #     Tfee = 0.000062  # The average transaction fee
    #     Tsize = 0.000546  # The average transaction size  in MB
    #
    #     ''' Node Parameters '''
    #     NUM_NODES = 15  # the total number of nodes in the network
    #     NODES = []
    #     from Models.Node import Node
    #     # here as an example we define three nodes by assigning a unique id for each one
    #     NODES = [Node(id=0, hashPower=50), Node(id=1, hashPower=0), Node(id=2, hashPower=0),
    #              Node(id=3, hashPower=150), Node(id=4, hashPower=50), Node(id=5, hashPower=150),
    #              Node(id=6, hashPower=0), Node(id=7, hashPower=100), Node(id=8, hashPower=0),
    #              Node(id=9, hashPower=0),Node(id=10, hashPower=0), Node(id=11, hashPower=0),
    #              Node(id=12, hashPower=0), Node(id=13, hashPower=0), Node(id=14, hashPower=100)]
    #
    #     ''' Simulation Parameters '''
    #     simTime = 100000  # the simulation length (in seconds)
    #     Runs = 1  # Number of simulation runs

    ''' Input configurations for Bitcoin model '''
    if model == 1:
        ''' Block Parameters '''
        Binterval = 600  # Average time (in seconds)for creating a block in the blockchain
        Bsize = 1.0  # The block size in MB
        Bdelay = 0.42  # average block propogation delay in seconds, #Ref: https://bitslog.wordpress.com/2016/04/28/uncle-mining-an-ethereum-consensus-protocol-flaw/
        Breward = 12.5  # Reward for mining a block
        Rreward = 0.09  # Reward for redacting a transaction

        ''' Transaction Parameters '''
        hasTrans = True  # True/False to enable/disable transactions in the simulator
        Ttechnique = "Light"  # Full/Light to specify the way of modelling transactions
        Tn = 5  # The rate of the number of transactions to be created per second

        Tdelay = 5.1 # The average transaction propagation delay in seconds (Only if Full technique is used)
        Tfee = 0.001  # The average transaction fee
        Tsize = 0.0006  # The average transaction size in MB

        ''' Node Parameters '''
        NUM_NODES = 100  # the total number of nodes in the network
        NODES = []
        MINERS_PORTION = 0.27 # Example: 0.5 ==> 50% of miners
        MAX_HASH_POWER = 200
        from Models.Bitcoin.Node import Node
        num_miners = int(NUM_NODES * MINERS_PORTION)

        # Create miners
        for i in range(num_miners):
            hash_power = random.randint(1, MAX_HASH_POWER)
            NODES.append(Node(id=i, hashPower=hash_power))
        # Create regular nodes
        for i in range(num_miners, NUM_NODES):
            NODES.append(Node(id=i, hashPower=0))

        ''' Simulation Parameters '''
        simTime = 40000# the simulation length (in seconds)
        Runs = 1  # Number of simulation runs

        ''' Redaction Parameters'''
        hasRedact = True
        hasMulti = True
        redactRuns = 3
        adminNode = random.randint(0, len(NODES))
        editExperiment = True # 开启阶段2实验
        # adminNode = 50

        # === Redactable‑vs‑Rebuild 实验参数 ===
        editIntervalSec = 960  # 每 120秒触发一次更新需求
        updateTxSizeMB = 0.0006  # 更新交易大小；可用原 Tsize

