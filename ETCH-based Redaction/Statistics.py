from InputsConfig import InputsConfig as p
from Models.Consensus import Consensus as c
from Models.Incentives import Incentives
import pandas as pd
import os
from openpyxl import load_workbook
import matplotlib.pyplot as plt
import numpy as np
class Statistics:
    # Global variables used to calculate and print stimulation results
    totalBlocks = 0
    mainBlocks = 0
    staleBlocks = 0
    staleRate = 0
    blockData = []
    blocksResults = []
    blocksSize = []
    profits = [[0 for x in range(7)] for y in
               range(p.Runs * len(p.NODES))]  # rows number of miners * number of runs, columns =7
    index = 0
    original_chain = []
    chain = []
    redactResults = []
    allRedactRuns = []
    round = 0

    def calculate(t):
        Statistics.global_chain()  # print the global chain
        Statistics.blocks_results(t)  # calculate and print block statistics e.g., # of accepted blocks and stale rate etc
        if p.hasRedact:
            Statistics.redact_result()  # to calculate the info per redact operation

    # Calculate block statistics Results
    def blocks_results(t):
        trans = 0
        Statistics.mainBlocks = len(c.global_chain) - 1
        Statistics.staleBlocks = Statistics.totalBlocks - Statistics.mainBlocks
        for b in c.global_chain:
            trans += len(b.transactions)
        Statistics.staleRate = round(Statistics.staleBlocks / Statistics.totalBlocks * 100, 2)
        Statistics.blockData = [Statistics.totalBlocks, Statistics.mainBlocks, Statistics.staleBlocks, Statistics.staleRate, trans, t, str(Statistics.blocksSize)]
        Statistics.blocksResults += [Statistics.blockData]

    ############################ Calculate and distibute rewards among the miners #############################
    def profit_results(self):

        for m in p.NODES:
            i = Statistics.index + m.id * p.Runs
            Statistics.profits[i][0] = m.id
            Statistics.profits[i][1] = m.hashPower
            Statistics.profits[i][2] = m.blocks
            Statistics.profits[i][3] = round(m.blocks / Statistics.mainBlocks * 100, 2)
            Statistics.profits[i][4] = 0
            Statistics.profits[i][5] = 0
            Statistics.profits[i][6] = m.balance
        #print("Profits :")
        #print(Statistics.profits)

        Statistics.index += 1

    ########################################################### prepare the global chain  ###########################################################################################
    def global_chain():
        for i in c.global_chain:
            block = [i.depth, i.id, i.previous, i.timestamp, i.miner, len(i.transactions), i.size]
            Statistics.chain += [block]
        print("Length of CHAIN = "+str(len(Statistics.chain)))
        # print(Statistics.chain)


    def original_global_chain():
        for i in c.global_chain:
            block = [i.depth, i.id, i.previous, i.timestamp, i.miner, len(i.transactions), str(i.size)]
            Statistics.original_chain += [block]


    ########################################################## generate redaction data ############################################################
    def redact_result():
        i = 0
        profit_count, op_count = 0, p.redactRuns
        while i < len(p.NODES):
            if p.redactRuns == 0:
                profit_count = 0
            if len(p.NODES[i].redacted_tx) != 0 and p.redactRuns > 0:
                for j in range(len(p.NODES[i].redacted_tx)):
                    print(f'Deletion/Redaction: Block Depth => {p.NODES[i].redacted_tx[j][0]}, Transaction ID => {p.NODES[i].redacted_tx[j][1].id}')
                    # Added Miner ID,Block Depth,Transaction ID,Redaction Profit,Performance Time (ms),Blockchain Length,# of Tx
                    result = [p.NODES[i].id, p.NODES[i].redacted_tx[j][0], p.NODES[i].redacted_tx[j][1].id,
                              p.NODES[i].redacted_tx[j][2], p.NODES[i].redacted_tx[j][3],
                              p.NODES[i].redacted_tx[j][4], p.NODES[i].redacted_tx[j][5]]
                    profit_count += p.NODES[i].redacted_tx[j][2]
                    Statistics.redactResults.append(result)
            i += 1
        Statistics.allRedactRuns.append([profit_count, op_count])

    ########################################################### Print simulation results to Excel ###########################################################################################
    def print_to_excel(fname):

        df1 = pd.DataFrame(
            {'Block Time': [p.Binterval], 'Block Propagation Delay': [p.Bdelay], 'No. Miners': [len(p.NODES)],
             'Simulation Time': [p.simTime]})
        # data = {'Stale Rate': Results.staleRate,'# Stale Blocks': Results.staleBlocks,'# Total Blocks': Results.totalBlocks, '# Included Blocks': Results.mainBlocks}

        df2 = pd.DataFrame(Statistics.blocksResults)
        df2.columns = ['Total Blocks', 'Main Blocks', 'Stale Blocks', 'Stale Rate',
                       '# transactions', 'Performance Time', 'Block sizeeeeeee']

        # df3 = pd.DataFrame(Statistics.profits)
        # df3.columns = ['Miner ID', '% Hash Power', '# Mined Blocks', '% of main blocks', '# Uncle Blocks',
        #  '% of uncles', 'Profit (in ETH)']

        df4 = pd.DataFrame(Statistics.chain)
        print(df4)
        # df4.columns= ['Block Depth', 'Block ID', 'Previous Block', 'Block Timestamp', 'Miner ID', '# transactions','Block Size']
        df4.columns = ['Block Depth', 'Block ID', 'Previous Block', 'Block Timestamp', 'Miner ID', '# transactions',
                           'Block Size']

        if p.hasRedact:
            if p.redactRuns > 0:
                # blockchain history before redaction
                df7 = pd.DataFrame(Statistics.original_chain)
                # df4.columns= ['Block Depth', 'Block ID', 'Previous Block', 'Block Timestamp', 'Miner ID', '# transactions','Block Size']
                df7.columns = ['Block Depth', 'Block ID', 'Previous Block', 'Block Timestamp', 'Miner ID', '# transactions', 'Block Size']

                # Redaction results
                df5 = pd.DataFrame(Statistics.redactResults)
                print(df5)
                df5.columns = ['Miner ID', 'Block Depth', 'Transaction ID', 'Redaction Profit', 'Performance Time (ms)', 'Blockchain Length', '# of Tx']

            df6 = pd.DataFrame(Statistics.allRedactRuns)
            print(df6)
            df6.columns = ['Total Profit/Cost', 'Redact op runs']
        writer = pd.ExcelWriter(fname, engine='xlsxwriter')
        df1.to_excel(writer, sheet_name='InputConfig')
        df2.to_excel(writer, sheet_name='SimOutput')
        # df3.to_excel(writer, sheet_name='Profit')
        if p.hasRedact and p.redactRuns > 0:
            df2.to_csv('Results/time_redact.csv', sep=',', mode='a+', index=False, header=False, encoding='utf-8')
            df7.to_excel(writer, sheet_name='ChainBeforeRedaction')
            df5.to_excel(writer, sheet_name='RedactResult')
            df4.to_excel(writer, sheet_name='Chain')
            # Add the result to transaction/performance time csv to statistic analysis
            # df5.to_csv('Results_new/tx_time.csv', sep=',', mode='a+', index=False, header=False,encoding='utf-8')
            # Add the result to block length/performance time csv to statistic analysis, and fixed the number of transactions
            df5.to_csv('Results/block_time.csv', sep=',', mode='a+', index=False, header=False,encoding='utf-8')
            if p.hasMulti:
                df5.to_csv('Results/block_time_den.csv', sep=',', mode='a+', index=False, header=False)
                df5.to_csv('Results/tx_time_den.csv', sep=',', mode='a+', index=False, header=False)
            # Add the total profit earned vs the number of redaction operation runs
            df6.to_csv('Results/profit_redactRuns.csv', sep=',', mode='a+', index=False, header=False)
        else:
            df4.to_excel(writer, sheet_name='Chain')
            df2.to_csv('Results/time.csv', sep=',', mode='a+', index=False, header=False)
        writer._save()


    ########################################################### Reset all global variables used to calculate the simulation results ###########################################################################################
    def reset():
        Statistics.totalBlocks = 0
        Statistics.mainBlocks = 0
        Statistics.staleBlocks = 0
        Statistics.staleRate = 0
        Statistics.blockData = []

    def reset2():
        Statistics.blocksResults = []
        Statistics.profits = [[0 for x in range(7)] for y in
                              range(p.Runs * len(p.NODES))]  # rows number of miners * number of runs, columns =7
        Statistics.index = 0
        Statistics.chain = []
        Statistics.redactResults = []
        Statistics.allRedactRuns = []

     # === 实验阶段2: 编辑 vs 重建的统计 ===
    etch_edit_time_list = []  # 每次编辑的耗时 (ms)
    rebuild_time_list = []  # 每次重建新区块的耗时 (ms)
    etch_chain_size_list = []  # 每次编辑后链尺寸 (MB)
    rebuild_chain_size_list = []  # 每次重建后链尺寸 (MB)
    etch_edit_time_ts = []  # 编辑耗时时间戳（用于随时间绘图）
    rebuild_time_ts = []  # 重建耗时时间戳
    etch_chain_size_ts = []  # 编辑后链尺寸随时间
    rebuild_chain_size_ts = []  # 重建后链尺寸随时间

    @staticmethod
    def print_update_stats():
        if Statistics.etch_edit_time_list:
            print(
                f"[ETCH] 平均编辑耗时: {sum(Statistics.etch_edit_time_list) / len(Statistics.etch_edit_time_list):.2f} ms")
            print(f"[ETCH] 当前链尺寸: {Statistics.etch_chain_size_list[-1]:.4f} MB")
        if Statistics.rebuild_time_list:
            print(
                f"[Rebuild] 平均新区块耗时: {sum(Statistics.rebuild_time_list) / len(Statistics.rebuild_time_list):.2f} ms")
            print(f"[Rebuild] 当前链尺寸: {Statistics.rebuild_chain_size_list[-1]:.4f} MB")


    '''
    @staticmethod
    def plot_update_stats():
        #if Statistics.etch_edit_time_ts:
            plt.figure()
            plt.plot(Statistics.etch_edit_time_ts, Statistics.etch_edit_time_list, label="ETCH Edit Time (ms)")
            plt.plot(Statistics.rebuild_time_ts, Statistics.rebuild_time_list, label="Rebuild Time (ms)")
            plt.xlabel("Sim Time")
            plt.ylabel("Time (ms)")
            plt.legend()
            plt.title("Edit vs Rebuild - Time Cost")
            plt.savefig("edit_vs_rebuild_time.png")
            plt.show()
        #if Statistics.etch_chain_size_ts:
            plt.figure()
            plt.plot(Statistics.etch_chain_size_ts, Statistics.etch_chain_size_list, label="ETCH Chain Size (MB)")
            plt.plot(Statistics.rebuild_chain_size_ts, Statistics.rebuild_chain_size_list,
                     label="Rebuild Chain Size (MB)")
            plt.xlabel("Sim Time")
            plt.ylabel("Chain Size (MB)")
            plt.legend()
            plt.title("Edit vs Rebuild - Chain Size Growth")
            plt.savefig("edit_vs_rebuild_chain_size.png")
            plt.show()
    '''

    def save_ETCH_simulation_to_excel():
        data = {
            'Time (ms)': Statistics.etch_edit_time_ts,
            'Chain Length': Statistics.etch_chain_size_list,
            'Edit Duration (ms)': Statistics.etch_edit_time_list
        }
        filename = "etch-output.xlsx"
        df = pd.DataFrame(data)
        df.to_excel(filename, index=False)
        print(f"Saved simulation data to {filename}")

    def save_Rebuild_simulation_to_excel():
        data = {
            'Time (ms)': Statistics.rebuild_time_ts,
            'Chain Length': Statistics.rebuild_chain_size_list,
            'Rebuild Duration (ms)': Statistics.rebuild_time_list
        }
        filename = "Rebuild-output.xlsx"
        df = pd.DataFrame(data)
        df.to_excel(filename, index=False)
        print(f"Saved simulation data to {filename}")

    def plot_update_stats():
        """
        在同一张图中对比 ETCH 编辑 vs Rebuild 的：
        1. 时间开销
        2. 链尺寸增长
        """

        # === 图1：时间开销对比 ===
        if Statistics.etch_edit_time_ts and Statistics.rebuild_time_ts:
            plt.figure(figsize=(10, 5))
            plt.plot(Statistics.etch_edit_time_ts, Statistics.etch_edit_time_list,
                     label="ETCH Edit Time", color='blue', linestyle='-', marker='o')
            plt.plot(Statistics.rebuild_time_ts, Statistics.rebuild_time_list,
                     label="Rebuild Time", color='red', linestyle='--', marker='x')
            plt.xlabel("Simulation Time")
            plt.ylabel("Time Cost (ms)")
            plt.title("Edit vs Rebuild - Time Overhead")
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.5)
            plt.tight_layout()
            plt.savefig("edit_vs_rebuild_time.png", dpi=300)
            plt.show()
        else:
            print("⚠️ 时间数据为空，无法绘图。")

        # === 图2：链尺寸对比 ===
        if Statistics.etch_chain_size_ts and Statistics.rebuild_chain_size_ts:
            plt.figure(figsize=(10, 5))
            plt.plot(Statistics.etch_chain_size_ts, Statistics.etch_chain_size_list,
                     label="ETCH Chain Size", color='green', linestyle='-', marker='o')
            plt.plot(Statistics.rebuild_chain_size_ts, Statistics.rebuild_chain_size_list,
                     label="Rebuild Chain Size", color='orange', linestyle='--', marker='x')
            plt.xlabel("Simulation Time")
            plt.ylabel("Chain Size (MB)")
            plt.title("Edit vs Rebuild - Chain Size Growth")
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.5)
            plt.tight_layout()
            plt.savefig("edit_vs_rebuild_chain_size.png", dpi=300)
            plt.show()
        else:
            print("⚠️ 链尺寸数据为空，无法绘图。")
