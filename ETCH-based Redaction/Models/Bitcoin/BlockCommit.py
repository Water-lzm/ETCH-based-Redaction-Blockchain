import hashlib
import json
import random
import time
import CH.ChameleonHash as ch
import CH.SecretSharing as ss
import CH.DTCH
from CH.DTCH import participants_forge_Prepare, participants_forge, DTCH_verify, DCH_Hash, participant_PreparePoly, \
    participant_check
from CH.ChameleonHash import q, g, SK, PK, KeyGen, forge, forgeSplit, chameleonHash, chameleonHashSplit
from InputsConfig import InputsConfig as p
from Models.Bitcoin.Consensus import Consensus as c
from Models.BlockCommit import BlockCommit as BaseBlockCommit
from Models.Network import Network
from Models.Transaction import Transaction as Tr, LightTransaction as LT, FullTransaction as FT
from Scheduler import Scheduler
from Statistics import Statistics
import concurrent.futures
import matplotlib.pyplot as plt
from Models.Block import Block
from Event import Event, Queue
import copy

pa = 18604511303632357477261733749289932684042548414204891841229696446591
SKlist = []
PKlist = []
shares = []
rlist = []

# 在类开始部分添加全局计数器和计时数据列表
block_counter = 0
block_times = []
hash_times = []
# 实验统计用列表
redactionTimes = []


class Participant:  # 首字母大写
    def __init__(self, node_id, hash_power, p, q, g):
        self.node_id = node_id
        self.hash_power = hash_power
        self.p = p
        self.q = q
        self.g = g


class BlockCommit(BaseBlockCommit):
    edit_success_count = 0
    edit_failure_count = 0
    edit_latencies = []
    # 用于TPS评估
    total_tx_count = 0
    total_sim_time = 0.0

    @staticmethod
    def record_redaction_time(t):
        redactionTimes.append(t)

    @staticmethod
    def summarize_redaction_times():
        if redactionTimes:
            avg = sum(redactionTimes) / len(redactionTimes)
            print(
                f"[Stats] Avg Edit Time: {avg:.3f} ms, Max: {max(redactionTimes):.3f} ms, Min: {min(redactionTimes):.3f} ms")

    @staticmethod
    def redact_tx_with_stats(miner, block_i, tx_i, fee, delay_ms=0):
        try:
            latency = BlockCommit.redact_tx(miner, block_i, tx_i, fee, delay_ms)
            BlockCommit.edit_success_count += 1
            BlockCommit.edit_latencies.append(latency)
            return True
        except Exception as e:
            BlockCommit.edit_failure_count += 1
            print(f"[Edit Failed] Miner {miner.id} at Block {block_i}, Tx {tx_i}: {str(e)}")
            return False

    @staticmethod
    def redact_tx(miner, i, tx_i, fee, delay_ms):
        block = miner.blockchain[i]
        x1 = json.dumps([[i.id for i in block.transactions], block.previous], sort_keys=True).encode()
        m1 = hashlib.sha256(x1).hexdigest()

        block.transactions[tx_i].fee = fee
        block.transactions[tx_i].id = random.randrange(100000000000)
        miner.redacted_tx.append(
            [i, block.transactions[tx_i], 0, 0, miner.blockchain_length(), len(block.transactions)])

        x2 = json.dumps([[i.id for i in block.transactions], block.previous], sort_keys=True).encode()
        m2 = hashlib.sha256(x2).hexdigest()
        if p.hasMulti:
            miner_list = [miner for miner in p.NODES if miner.hashPower > 0]
            t = len(miner_list)
            miner_id_list = [miner.id for miner in miner_list]
            id_list = [random.randint(1, 100) for _ in range(t)]

            ts_KGen = time.perf_counter()
            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = [executor.submit(participant_PreparePoly, id_list) for _ in miner_id_list]
                results = [f.result() for f in concurrent.futures.as_completed(futures)]
            C = [results[i][0] for i in range(t)]
            F_ji = [[results[k][1][i] for k in range(t)] for i in range(t)]

            with concurrent.futures.ThreadPoolExecutor() as executor:
                verification_futures = [executor.submit(participant_check, id, f_ji, C, id_list)
                                        for id, f_ji in zip(id_list, F_ji)]
                verification_results = [f.result() for f in concurrent.futures.as_completed(verification_futures)]

            te_KGen = time.perf_counter()
            hk = verification_results[0][2]  # 系统公钥
            sk_shares = [r[0] for r in verification_results]

            h, r = DCH_Hash(hk, m1)

            t1 = time.time()
            if delay_ms > 0:
                time.sleep(delay_ms / 1000.0)  # 模拟网络延迟
            with concurrent.futures.ThreadPoolExecutor() as executor:
                Forge_PrePare_results = [f.result() for f in
                                         concurrent.futures.as_completed([
                                             executor.submit(participants_forge_Prepare, id)
                                             for id in id_list
                                         ])]

            k_i = [r[0] for r in Forge_PrePare_results]
            K = 1
            for r in Forge_PrePare_results:
                K = (K * r[1]) % pa

            with concurrent.futures.ThreadPoolExecutor() as executor:
                Forge_results = [f.result() for f in
                                 concurrent.futures.as_completed([
                                     executor.submit(participants_forge, id, K, ki, si, m2, h, id_list)
                                     for id, ki, si in zip(id_list, k_i, sk_shares)
                                 ])]

            r0 = Forge_results[0][0]
            w_prime = sum(r[1] for r in Forge_results) % q
            r_prime = (r0, w_prime)

            id2 = DCH_Hash(hk, m2)[0]
            block.r = r_prime
            for node in p.NODES:
                block_index = len(node.blockchain) - 1
                if node.id != miner.id:
                    node.blockchain[block_index].transactions = block.transactions
                    node.blockchain[block_index].r = block.r
        else:
            r2 = forge(SK, m1, block.r, m2)
            id2 = chameleonHash(PK, m2, r2)
            block.r = r2

        block.id = id2
        t2 = time.time()
        t = (t2 - t1) * 1000
        BlockCommit.record_redaction_time(t)
        print(f"[Edit] Miner {miner.id} edit time = {t:.3f} ms")
        return t

    @staticmethod
    def generate_edit_experiment():
        '''
        print("\n[Exp2A] 单次编辑耗时评估:")
        for _ in range(10):
            miner = random.choice([node for node in p.NODES if node.hashPower > 0])
            blk_i = random.randint(1, len(miner.blockchain) - 1)
            tx_i = random.randint(0, len(miner.blockchain[blk_i].transactions) - 1)
            BlockCommit.redact_tx(miner, blk_i, tx_i, p.Tfee)
        BlockCommit.summarize_redaction_times()

        print("\n[Exp2B] 延迟模拟 (RTT) 编辑时间影响:")
        for rtt in [0, 20, 50, 100, 200, 300]:
            redactionTimes.clear()
            print(f"-- 模拟 RTT = {rtt} ms")
            for _ in range(5):
                miner = random.choice([node for node in p.NODES if node.hashPower > 0])
                blk_i = random.randint(1, len(miner.blockchain) - 1)
                tx_i = random.randint(0, len(miner.blockchain[blk_i].transactions) - 1)
                BlockCommit.redact_tx(miner, blk_i, tx_i, p.Tfee, delay_ms=rtt)
            BlockCommit.summarize_redaction_times()
        '''

    # Handling and running Events
    def handle_event(event):
        if event.type == "create_block":
            print('执行创建区块事件')
            BlockCommit.generate_block(event)
        elif event.type == "receive_block":
            BlockCommit.receive_block(event)
        elif event.type == "update_request":
            print('执行更新区块事件')
            BlockCommit.handle_update_request(event)

    # Block Creation Event
    def generate_block(event):
        global block_counter, block_times, hash_times
        miner = p.NODES[event.block.miner]
        minerId = miner.id
        eventTime = event.time
        blockPrev = event.block.previous
        if blockPrev == miner.last_block().id or blockPrev != miner.last_block().id:
            Statistics.totalBlocks += 1  # count # of total blocks created!
            start_block_time = time.perf_counter()  # <<<< start
            if p.hasTrans:
                if p.Ttechnique == "Light":
                    blockTrans, blockSize = LT.execute_transactions()  # Get the created block (transactions and block size)
                    print('交易数：', len(blockTrans))
                    print('miner id:', minerId)
                    BlockCommit.total_tx_count += len(blockTrans)
                    Statistics.blocksSize = blockSize
                elif p.Ttechnique == "Full":
                    blockTrans, blockSize = FT.execute_transactions(miner, eventTime)

                event.block.transactions = blockTrans
                event.block.size = blockSize
                event.block.usedgas = blockSize

                # hash the transactions and previous hash value
                start_hash = time.perf_counter()
                if p.hasRedact:
                    event.block.r = random.randint(1, q)
                    x = json.dumps([[i.id for i in event.block.transactions], event.block.previous],
                                   sort_keys=True).encode()
                    m = hashlib.sha256(x).hexdigest()
                    event.block.id = DCH_Hash(miner.PK, m)[0]
                # 新增测试原始哈希创建区块的开销
                else:
                    x = json.dumps([[i.id for i in event.block.transactions], event.block.previous],
                                   sort_keys=True).encode()
                    event.block.id = hashlib.sha256(x).hexdigest()
                end_hash = time.perf_counter()
                hash_time = (end_hash - start_hash) * 1000
                hash_times.append(hash_time)
            miner.blockchain.append(event.block)

            BlockCommit.propagate_block(event.block)
            end_block_time = time.perf_counter()  # <<<< end
            block_time = (end_block_time - start_block_time) * 1000
            # print(f"[Block] Generated by miner {minerId}, time = {block_time:.3f} ms")
            block_times.append(block_time)
            block_counter += 1
            BlockCommit.generate_next_block(miner, eventTime)  # Start mining or working on the next block
            print('创建后chain size:', len(miner.blockchain))
            '''
            # 满 100 个区块后终止模拟
            if block_counter >= 100:
                avg_block_time = sum(block_times) / len(block_times)
                avg_hash_time = sum(hash_times) / len(hash_times)
                print('交易总数：',BlockCommit.total_tx_count)
                real_tps = BlockCommit.total_tx_count / (sum(block_times)/1000+(c.Protocol(miner)*len(block_times))-1)
                print("\n==== 区块生成统计 ====")
                print(f"总区块数: {block_counter}")
                print(f"平均区块生成时间: {avg_block_time:.3f} ms")
                print(f"平均哈希计算时间: {avg_hash_time:.3f} ms")
                print(f"[TPS] 替换哈希后系统实际处理交易吞吐: {real_tps:.2f} tx/sec")
                import sys
                sys.exit(0)
            '''
    # Block Receiving Event
    def receive_block(event):
        miner = p.NODES[event.block.miner]
        minerId = miner.id
        currentTime = event.time
        blockPrev = event.block.previous  # previous block id
        node = p.NODES[event.node]  # recipient
        lastBlockId = node.last_block().id  # the id of last block

        #### case 1: the received block is built on top of the last block according to the recipient's blockchain ####
        if blockPrev == lastBlockId:
            node.blockchain.append(event.block)  # append the block to local blockchain
            if p.hasTrans and p.Ttechnique == "Full":
                BlockCommit.update_transactionsPool(node, event.block)
            BlockCommit.generate_next_block(node, currentTime)  # Start mining or working on the next block

        #### case 2: the received block is not built on top of the last block ####
        else:
            depth = event.block.depth + 1
            if depth > len(node.blockchain):
                BlockCommit.update_local_blockchain(node, miner, depth)
                BlockCommit.generate_next_block(node, currentTime)  # Start mining or working on the next block

            if p.hasTrans and p.Ttechnique == "Full":
                BlockCommit.update_transactionsPool(node, event.block)  # not sure yet.

    # Upon generating or receiving a block, the miner start working on the next block as in POW
    def generate_next_block(node, currentTime):
        if node.hashPower > 0:
            blockTime = currentTime + c.Protocol(node)  # time when miner x generate the next block
            Scheduler.create_block_event(node, blockTime)

    def generate_initial_events():
        currentTime = 0
        for node in p.NODES:
            BlockCommit.generate_next_block(node, currentTime)
        BlockCommit._schedule_next_update(currentTime)

    def propagate_block(block):
        for recipient in p.NODES:
            if recipient.id != block.miner:
                blockDelay = Network.block_prop_delay()
                # draw block propagation delay from a distribution !! or assign 0 to ignore block propagation delay
                Scheduler.receive_block_event(recipient, block, blockDelay)
        return blockDelay

    def setupSecretSharing():
        global SKlist, PKlist, rlist, shares
        SKlist, PKlist = KeyGen(ch.p, q, g, len(p.NODES))
        rlist = ch.getr(len(p.NODES), q)
        for i, node in enumerate(p.NODES):
            node.PK = PKlist[i]
            node.SK = SKlist[i]

    def generate_redaction_event(redactRuns):
        t1 = time.time()
        i = 0
        miner_list = [node for node in p.NODES if node.hashPower > 0]
        while i < redactRuns:
            if p.hasMulti:
                miner = random.choice(miner_list)
            else:
                miner = p.NODES[p.adminNode]
            r = random.randint(1, 2)
            # r =2
            block_index = random.randint(1, len(miner.blockchain) - 1)
            tx_index = random.randint(1, len(miner.blockchain[block_index].transactions) - 1)
            if r == 1:
                BlockCommit.redact_tx(miner, block_index, tx_index, p.Tfee, 100)
            else:
                BlockCommit.delete_tx(miner, block_index, tx_index)
            t2 = time.time()
            t = (t2 - t1) * 1000  # in ms
            print(f"Redaction time = {t} ms")
            i += 1

    def delete_tx(miner, i, tx_i):
        t1 = time.time()
        block = miner.blockchain[i]
        # Store the old block data
        x1 = json.dumps([[i.id for i in block.transactions], block.previous], sort_keys=True).encode()
        m1 = hashlib.sha256(x1).hexdigest()

        # record the block index and deleted transaction object, miner reward  = 0 and performance time = 0
        # and also the blockchain size, number of transaction of that action block
        miner.redacted_tx.append(
            [i, block.transactions.pop(tx_i), 0, 0, miner.blockchain_length(), len(block.transactions)])

        # Store the modified block data
        x2 = json.dumps([[i.id for i in block.transactions], block.previous], sort_keys=True).encode()
        m2 = hashlib.sha256(x2).hexdigest()
        # Forge new r
        # t1 = time.time()
        if p.hasMulti:
            # rlist = block.r
            miner_list = [miner for miner in p.NODES if miner.hashPower > 0]
            t = len(miner_list)
            miner_id_list = [miner.id for miner in miner_list]
            # 每个参与者的ID
            id_list = [random.randint(1, 100) for _ in range(t)]
            # propagation delay in sharing secret key
            # time.sleep(0.005)
            # 测试Time of KGen
            ts_KGen = time.perf_counter()
            # 使用线程池并行化每个参与者的计算过程
            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = [executor.submit(participant_PreparePoly, id_list) for miner_id in miner_id_list]
                results = [future.result() for future in concurrent.futures.as_completed(futures)]
            C = [results[i][0] for i in range(t)]
            F_ji = []
            for i in range(t):
                F_ji.append([])
                for k in range(t):
                    F_ji[i].append(results[k][1][i])
            # print('f_ji for every i:',F_ji)
            # 使用线程池并行化每个参与者的承诺验证过程，每个参与者返回（si,pki,pk）
            with concurrent.futures.ThreadPoolExecutor() as executor:
                verification_futures = [executor.submit(participant_check, id, f_ji, C, id_list) for
                                        id, f_ji in zip(id_list, F_ji)]
                verification_results = [future.result() for future in
                                        concurrent.futures.as_completed(verification_futures)]
            # print('(s_i,pk_i,pk):',verification_results)
            te_KGen = time.perf_counter()
            hk = verification_results[0][2]  # 系统公钥
            for i in range(t):
                print('参与者' + str(id_list[i]) + '的最终私钥份额(si,pk_i):',
                      (verification_results[i][0], verification_results[i][1]))
            print('time of KGen(ms):', (te_KGen - ts_KGen) * 1000)

            # 准备Forge算法的形参sk_shares并按顺序传递
            sk_shares = []
            for i in range(t):
                sk_shares.append(verification_results[i][0])
            # print('sk_shares:',sk_shares)
            # 计算Time of Hash
            h, r = DCH_Hash(hk, m1)
            # id2 = h
            # print('h,(r,w):',h,r)

            # 测试Forge
            with concurrent.futures.ThreadPoolExecutor() as executor:
                Forge_PrePare_futures = [executor.submit(participants_forge_Prepare, id) for id in id_list]
                Forge_PrePare_results = [future.result() for future in
                                         concurrent.futures.as_completed(Forge_PrePare_futures)]

            # print('Forge_PrePare_results:',Forge_PrePare_results)

            k_i = [Forge_PrePare_results[i][0] for i in range(t)]
            K = 1
            for i in range(t):
                K *= Forge_PrePare_results[i][1] % pa
            K = K % pa

            with concurrent.futures.ThreadPoolExecutor() as executor:
                Forge_futures = [executor.submit(participants_forge, id, K, k_i, si, m2, h, id_list) for id, k_i, si in
                                 zip(id_list, k_i, sk_shares)]
                Forge_results = [future.result() for future in concurrent.futures.as_completed(Forge_futures)]
            r0 = Forge_results[0][0]
            w_prime = 0
            for i in range(t):
                w_prime += Forge_results[i][1]
            w_prime = w_prime % q
            r_prime = (r0, w_prime)

            id2 = DCH_Hash(hk, m2)[0]
            # 更新区块的变色龙哈希参数
            block.r = r_prime
            for node in p.NODES:
                block_index = len(node.blockchain) - 1  # 更清晰的变量名 新加的
                if node.id != miner.id:
                    # 安全修改
                    node.blockchain[block_index].transactions = block.transactions
                    node.blockchain[block_index].r = block.r
        else:
            r2 = forge(SK, m1, block.r, m2)
            id2 = chameleonHash(PK, m2, r2)
            block.r = r2
        t2 = time.time()
        block.id = id2
        # Calculate the performance time per operation
        # t2 = time.time()
        t = (t2 - t1) * 1000  # in ms
        # redact operation is more expensive than mining
        # print(f"Redaction succeeded in {t}")
        reward = random.expovariate(1 / p.Rreward)
        miner.balance += reward
        print('miner.redacted_tx:', miner.redacted_tx)
        print(f"Miner {miner.id} redacted_tx length: {len(miner.redacted_tx)}")
        print(f"Attempting to record reward: {reward}, time: {t}")
        miner.redacted_tx[-1][2] = reward
        miner.redacted_tx[-1][3] = t
        return miner

        #    def redact_tx(miner, i, tx_i, fee):
        t1 = time.time()
        block = miner.blockchain[i]
        # Store the old block data
        x1 = json.dumps([[i.id for i in block.transactions], block.previous], sort_keys=True).encode()
        m1 = hashlib.sha256(x1).hexdigest()

        # record the block depth and modify transaction information then recompute the transaction id
        block.transactions[tx_i].fee = fee
        block.transactions[tx_i].id = random.randrange(100000000000)
        # record the block depth, redacted transaction, miner reward = 0 and performance time = 0
        miner.redacted_tx.append(
            [i, block.transactions[tx_i], 0, 0, miner.blockchain_length(), len(block.transactions)])
        # Store the modified block data
        x2 = json.dumps([[i.id for i in block.transactions], block.previous], sort_keys=True).encode()
        m2 = hashlib.sha256(x2).hexdigest()
        # Forge new r
        # t1 = time.time()
        if p.hasMulti:
            miner_list = [miner for miner in p.NODES if miner.hashPower > 0]
            t = len(miner_list)
            miner_id_list = [miner.id for miner in miner_list]
            # 每个参与者的ID
            id_list = [random.randint(1, 100) for _ in range(t)]
            # propagation delay in sharing secret key
            # time.sleep(0.005)
            # 测试Time of KGen
            ts_KGen = time.perf_counter()
            # 使用线程池并行化每个参与者的计算过程
            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = [executor.submit(participant_PreparePoly, id_list) for miner_id in miner_id_list]
                results = [future.result() for future in concurrent.futures.as_completed(futures)]
            C = [results[i][0] for i in range(t)]
            F_ji = []
            for i in range(t):
                F_ji.append([])
                for k in range(t):
                    F_ji[i].append(results[k][1][i])
            # print('f_ji for every i:',F_ji)
            # 使用线程池并行化每个参与者的承诺验证过程，每个参与者返回（si,pki,pk）
            with concurrent.futures.ThreadPoolExecutor() as executor:
                verification_futures = [executor.submit(participant_check, id, f_ji, C, id_list) for
                                        id, f_ji in zip(id_list, F_ji)]
                verification_results = [future.result() for future in
                                        concurrent.futures.as_completed(verification_futures)]
            # print('(s_i,pk_i,pk):',verification_results)
            te_KGen = time.perf_counter()
            hk = verification_results[0][2]  # 系统公钥
            for i in range(t):
                print('参与者' + str(id_list[i]) + '的最终私钥份额(si,pk_i):',
                      (verification_results[i][0], verification_results[i][1]))
            print('time of KGen(ms):', (te_KGen - ts_KGen) * 1000)

            # 准备Forge算法的形参sk_shares并按顺序传递
            sk_shares = []
            for i in range(t):
                sk_shares.append(verification_results[i][0])
            # print('sk_shares:',sk_shares)
            # 计算Time of Hash
            h, r = DCH_Hash(hk, m1)
            # id2 = h
            # print('h,(r,w):',h,r)

            # 测试Forge
            with concurrent.futures.ThreadPoolExecutor() as executor:
                Forge_PrePare_futures = [executor.submit(participants_forge_Prepare, id) for id in id_list]
                Forge_PrePare_results = [future.result() for future in
                                         concurrent.futures.as_completed(Forge_PrePare_futures)]

            # print('Forge_PrePare_results:',Forge_PrePare_results)

            k_i = [Forge_PrePare_results[i][0] for i in range(t)]
            K = 1
            for i in range(t):
                K *= Forge_PrePare_results[i][1] % pa
            K = K % pa

            with concurrent.futures.ThreadPoolExecutor() as executor:
                Forge_futures = [executor.submit(participants_forge, id, K, k_i, si, m2, h, id_list) for id, k_i, si in
                                 zip(id_list, k_i, sk_shares)]
                Forge_results = [future.result() for future in concurrent.futures.as_completed(Forge_futures)]
            r0 = Forge_results[0][0]
            w_prime = 0
            for i in range(t):
                w_prime += Forge_results[i][1]
            w_prime = w_prime % q
            r_prime = (r0, w_prime)

            id2 = DCH_Hash(hk, m2)[0]
            # 更新区块的变色龙哈希参数
            block.r = r_prime
            for node in p.NODES:
                block_index = len(node.blockchain) - 1  # 更清晰的变量名 新加的
                if node.id != miner.id:
                    # 安全修改
                    node.blockchain[block_index].transactions = block.transactions
                    node.blockchain[block_index].r = block.r
        else:
            r2 = forge(SK, m1, block.r, m2)
            id2 = chameleonHash(PK, m2, r2)
            block.r = r2
        t2 = time.time()
        block.id = id2
        # Calculate the performance time per operation
        t = (t2 - t1) * 1000  # in ms
        # print(f"Redaction succeeded in {t}")
        # redact operation is more expensive than mining
        reward = random.expovariate(1 / p.Rreward)
        miner.balance += reward
        print(f"Miner {miner.id} redacted_tx length: {len(miner.redacted_tx)}")
        print(f"Attempting to record reward: {reward}, time: {t}")
        miner.redacted_tx[-1][2] = reward
        miner.redacted_tx[-1][3] = t
        return miner

    # ------------------------------------------------------------
    #  A. 生成“仅含更新交易”的新区块（重建方案用）
    # ------------------------------------------------------------
    @staticmethod
    def _create_update_block(tx, miner):
        """只打包 1 条更新交易，立即入链并广播。"""
        # 构造 1 条“更新交易”
        new_tx = tx
        prev = miner.blockchain[-1]
        blk = Block(previous=prev.id, depth=prev.depth + 1, miner=miner.id)
        blk.timestamp = time.time()
        blk.transactions = [new_tx]
        blk.size = random.expovariate(1 / p.Bsize)
        print('new creat block size:', blk.size)
        # 模拟交易处理耗时（可选，根据 Ttechnique 设置）
        if p.hasTrans:
            if p.Ttechnique == "Light":
                _, exec_size = LT.execute_transactions()
            elif p.Ttechnique == "Full":
                _, exec_size = FT.execute_transactions(miner, blk.timestamp)
            else:
                exec_size = blk.size  # fallback

            blk.usedgas = exec_size  # 仅作记录，无影响
        else:
            blk.usedgas = blk.size
        # 直接用 SHA‑256 计算块 ID（重建逻辑下不走 ETCH）
        digest = hashlib.sha256(json.dumps([[new_tx.id], prev.id], sort_keys=True).encode()).hexdigest()
        blk.id = digest

        miner.blockchain.append(blk)  # 本地入链
        blkdelay=BlockCommit.propagate_block(blk)  # 广播给其他节点
        print('blk delay:',blkdelay)
        return blkdelay

    # ------------------------------------------------------------
    #  B. 周期性“更新事件”调度器
    # ------------------------------------------------------------
    @staticmethod
    def _schedule_next_update(current_time):
        next_time = current_time + p.editIntervalSec
        Scheduler.custom_event("update_request", next_time)

    # ------------------------------------------------------------
    #  C. 处理 update_request 事件
    # ------------------------------------------------------------
    # @staticmethod

    '''
    def handle_update_request(event):
        """根据 hasRedact 决定走 ETCH 编辑还是 Rebuild 方案，并做统计。"""
        miner = p.NODES[p.adminNode]  # 由管理员节点发起即可

        # 1) 选择最近非空块和交易
        blk = None
        for b in reversed(miner.blockchain):
            if b.transactions:
                blk = b
                break
        if blk is None:
            print("[WARN] 当前链上没有任何包含交易的区块，跳过本轮编辑")
            BlockCommit._schedule_next_update(event.time)
            return
        else:
            print('正常对交易执行编辑')
        tx = random.choice(blk.transactions)

        # ------ ETCH 编辑 ------
        if p.hasRedact:
            t0 = time.perf_counter()
            BlockCommit.redact_tx(miner, blk.depth, blk.transactions.index(tx), fee=p.Tfee)
            elapsed = (time.perf_counter() - t0) * 1000
            chain_size_mb = sum(b.size for b in miner.blockchain)
            # 记录编辑耗时和链尺寸（MB）
            Statistics.etch_edit_time_list.append(elapsed)
            Statistics.etch_chain_size_list.append(chain_size_mb)
            Statistics.etch_edit_time_ts.append(event.time)
            Statistics.etch_chain_size_ts.append(event.time)
        # ------ Rebuild ------
        else:
            t0 = time.perf_counter()
            new_tx = Tr(tx.id,tx.sender,tx.to,tx.value,tx.size,tx.fee)
            new_tx.value = str(tx.value) + " [UPDATED]"
            new_tx.size = random.expovariate(1 / p.Tsize)
            BlockCommit._create_update_block(new_tx, miner)
            elapsed = (time.perf_counter() - t0) * 1000
            chain_size_mb = sum(b.size for b in miner.blockchain)
            # 记录重建耗时和链尺寸（MB）
            Statistics.rebuild_time_list.append(elapsed)
            Statistics.rebuild_chain_size_list.append(chain_size_mb)
            Statistics.rebuild_time_ts.append(event.time)
            Statistics.rebuild_chain_size_ts.append(event.time)
            print('更新后chain size:',len(p.NODES[p.adminNode].blockchain))
        Statistics.plot_update_stats()
        # 继续调度下一次更新
        BlockCommit._schedule_next_update(event.time)
        '''

    @staticmethod
    def handle_update_request(event):
        """对同一交易执行 ETCH 和 Rebuild 两种编辑路径，用于性能对比"""
        miner = p.NODES[p.adminNode]

        # 1) 选择最近非空块和交易
        blk = None
        for b in reversed(miner.blockchain):
            if b.transactions:
                blk = b
                break
        if blk is None:
            print("[WARN] 当前链上没有任何包含交易的区块，跳过本轮编辑")
            BlockCommit._schedule_next_update(event.time)
            return

        tx = random.choice(blk.transactions)
        # 拷贝当前链用于ETCH编辑
        '''
        blk_index = random.randint(1, len(miner.blockchain) - 1)
        etch_elapsed = BlockCommit.redact_tx(miner, blk_index, blk.transactions.index(tx), fee=p.Tfee, delay_ms=100)
        etch_chain_size = sum(b.size for b in miner.blockchain)
        ###### 3. 记录对比结果 ######
        Statistics.etch_edit_time_list.append(etch_elapsed+Network.block_prop_delay()*1000)
        Statistics.etch_chain_size_list.append(etch_chain_size)
        Statistics.etch_edit_time_ts.append(event.time)
        Statistics.etch_chain_size_ts.append(event.time)
        '''
        ###### 2. Rebuild 新区块 ######
        t1 = time.perf_counter()
        new_tx = Tr(tx.id, tx.sender, tx.to, tx.value, tx.size, tx.fee)
        new_tx.value = str(tx.value) + " [UPDATED]"
        new_tx.size = random.expovariate(1 / p.Tsize)
        delayt = BlockCommit._create_update_block(new_tx, miner)
        rebuild_elapsed = (time.perf_counter() - t1 + delayt) * 1000
        rebuild_chain_size = sum(b.size for b in miner.blockchain)
        Statistics.rebuild_time_list.append(rebuild_elapsed)
        Statistics.rebuild_chain_size_list.append(rebuild_chain_size)
        Statistics.rebuild_time_ts.append(event.time)
        Statistics.rebuild_chain_size_ts.append(event.time)

        #print(f"[Compare] t={event.time:.2f}s | ETCH: {etch_elapsed:.2f}ms, {etch_chain_size:.2f}MB")

        print(f"[Compare] t={event.time:.2f}s | Rebuild: {rebuild_elapsed:.2f}ms, {rebuild_chain_size:.2f}MB")
        # 继续调度下一次编辑事件
        BlockCommit._schedule_next_update(event.time)
