from InputsConfig import InputsConfig as p
from Event import Event, Queue
from Scheduler import Scheduler
from Statistics import Statistics
import time
from Models.Bitcoin.BlockCommit import BlockCommit
from Models.Bitcoin.Consensus import Consensus
from Models.Transaction import LightTransaction as LT, FullTransaction as FT
from Models.Bitcoin.Node import Node
from Models.Incentives import Incentives
import xlsxwriter

########################################################## Start Simulation ##############################################################


def main():
    print("START SIMULATION >>")
    for i in range(p.Runs):
        t1 = time.time()
        clock = 0  # set clock to 0 at the start of the simulation
        if p.hasTrans:
            if p.Ttechnique == "Light":
                LT.create_transactions()  # generate pending transactions
            elif p.Ttechnique == "Full":
                FT.create_transactions()  # generate pending transactions

        # if has multiplayer in the secret sharing
        # if p.hasMulti:
        #     BlockCommit.setupSecretSharing()
        #     for i in p.NODES:
        #         print(f'NODE {i.id}: Public Key: {i.PK}, Secret Key: {i.SK}')

        Node.generate_genesis_block()  # generate the genesis block for all miners
        # initiate initial events >= 1 to start with
        BlockCommit.generate_initial_events()

        while not Queue.isEmpty() and clock <= p.simTime:
            next_event = Queue.get_next_event()
            clock = next_event.time  # move clock to the time of the event
            BlockCommit.handle_event(next_event)
            Queue.remove_event(next_event)
        # test for chameleon hash forging

        if p.hasRedact:
            Consensus.fork_resolution()
            Statistics.original_global_chain()
            BlockCommit.generate_redaction_event(p.redactRuns)

        Consensus.fork_resolution()  # apply the longest chain to resolve the forks
        Incentives.distribute_rewards()  # distribute the rewards between the participating nodes
        t2 = time.time()
        t = (t2 -t1)* 1000
        print(f"Total time = {t}")

        # calculate the simulation results (e.g., block statistics and miners' rewards)
        Statistics.calculate(t)
        '''
        # === Phase 2: Editing Evaluation Experiments ===
        if hasattr(p, "editExperiment") and p.editExperiment:
            BlockCommit.generate_edit_experiment()
        '''

        ########## reset all global variable before the next run #############
        Statistics.reset()  # reset all variables used to calculate the results
        Node.resetState()  # reset all the states (blockchains) for all nodes in the network
        fname = "{0}M_{1}_{2}K.xlsx".format(p.Bsize / 1000000, p.Tn / 1000, p.Tsize)
        # print all the simulation results in an excel file
        Statistics.print_to_excel(fname)
        Statistics.reset2()  # reset profit results
        # === Reset all state for next run ===
        Statistics.reset() #阶段2实验新增
        Node.resetState() #阶段2实验新增

    print("\n==== SIMULATION ENDED ====")
    #Statistics.plot_update_stats()
    Statistics.save_ETCH_simulation_to_excel()
    Statistics.save_Rebuild_simulation_to_excel()
######################################################## Run Main method #####################################################################
if __name__ == '__main__':
    main()
