import random
import time
import hashlib
from InputsConfig import InputsConfig as p
#from pyunit_prime import get_large_prime_length  # 随机生成指定长度大素数
#from pyunit_prime import is_prime  # 判断素数
#from pyunit_prime import prime_range  # 输出指定区间素数
import math
import concurrent.futures

p = 18604511303632357477261733749289932684042548414204891841229696446591
q = 2238810024504495484628367478855587567273471529988554974877219789
g = 12340
#miner_list = [miner for miner in p.NODES if miner.hashPower > 0] degree of polynomial
t=3
def primeFactorization(length):  # 分解质因数
    global p, q
    #q = get_large_prime_length(length)
    q = 2238810024504495484628367478855587567273471529988554974877219789
    while True:
        d = random.randint(2, 10000)
        if d % 2 == 0:
            p = q * d + 1
            #if is_prime(p) == True:
            #    break
            #else:
            #    continue
        else:
            continue
   # primeList = prime_range(2, int(math.sqrt(d)))
   # result = [[0, 0] for i in range(len(primeList))]
   # for i in range(len(primeList)):
   #     result[i][0] = primeList[i]
   #     while d % primeList[i] == 0:
    #        result[i][1] += 1
    #        d = d // primeList[i]
    if d != 1:
        result.append([d, 1])
    result.append([q, 1])
    return result


def quickPower(a, b, c):  # 快速幂
    result = 1
    while b > 0:
        if b % 2 == 1:
            result = result * a % c
        a = a * a % c
        b >>= 1
    return result


def getGenerator(result):  # get g
    generator = random.randint(1, 1000)
    while True:
        if quickPower(generator, q, p) != 1:
            generator += 1
        else:
            for i in range(len(result)):
                if quickPower(generator, int((p - 1) / result[i][0]), p) == 1:
                    break
            if i != len(result) - 1:
                generator += 1
            else:
                break
    return generator
def evaluate_polynomial(coffs, x, q):
    result = 0
    for i in range(len(coffs)):
        result += coffs[i] * (x ** i)
        result %= q
    return result

def participant_PreparePoly(id_list):
    # 选择t-1次多项式的系数（包括私钥份额a_0）
    coefficients = [random.randint(1, q-1) for _ in range(t)]
    #打印多项式系数【a0,a1,a2】
    # #print("coeff:",coefficients)
    # 计算其他参与者的份额
    C_i=[quickPower(g,coefficients[i],p) for i in range(t)]
    #计算f_i(j)
    f_ij = [evaluate_polynomial(coefficients, id, q) for id in id_list]
    #print('C_i in prepare:',C_i)
    #print('f_ij:',f_ij)
    return C_i,f_ij


def participant_check(id,f_ji,C,id_list):
    #打印id_list,results
    #print('id in check:',id)
    #print('f_ji in check:', f_ji)
    #print('C:',C)
    for i in range(t):
        G_j = 1
        if id_list[i] != id:
            for j in range(t):
                G_j*=quickPower(C[i][j],quickPower(id,j,q),p)
            G_j=G_j % p
            if G_j!=quickPower(g,f_ji[i],p):
                print("Check False!")
                break
#    print("Check Success!")

    si=0
    pk=1
    for i in range(t):
        si+=f_ji[i]
        pk*=C[i][0]
    si=si%q
    pk = pk % p
    #print('ID-SI:',id,si)
    pk_i=quickPower(g,si,p)
    return si,pk_i,pk

def H0(hk,m):  # Huang_T0
    # 将 CID 和 hk 连接成一个字符串
    combined_string =hashlib.sha256((m + str(hk)).encode()).digest()
    combined_value=int.from_bytes(combined_string) % q
    # 将字符串转换为群元素
    group_element = quickPower(g, combined_value, p)
    # 假设 q 是群的阶数
    return group_element

def H1(m):
    # 计算 SHA-256 哈希值
    sha256_hash = hashlib.sha256(m.encode()).digest()
    # 将哈希值的最右侧的 k 位转换为整数，其中 k 是满足 2^k >= q 的最小整数
    integer_value = int.from_bytes(sha256_hash)
    # 返回 Z_q 中的元素
    return integer_value % q

def DCH_Hash(hk, m): #Huang_TCH
    r=random.randint(1,p-1)
    w = random.randint(1, q - 1)
    m_hash=m+str(r)
    e=H1(m_hash)
    h=(r*quickPower(hk,e,p)*quickPower(g,w,p))%p
    R=(r,w)
    return h,R

def participants_forge_Prepare(id):
    k_i=random.randint(1,q-1)
    K_i=quickPower(g,k_i,p)
    #print('k_i:',k_i)
    #print('K_i:',K_i)
    return k_i,K_i

def participants_forge(id,K,k_i,si,m_prime,h,id_list):
    #print('id in forge:',id)
    #print('r[0]:',r[0])
    r_prime=(h*quickPower(K,-1,p))%p
    m_hash = m_prime + str(r_prime)
    e_prime=H1(m_hash)
    lambda_i=1
    for j in range(t):
        if id_list[j] != id:
            lambda_i *= (id_list[j] * quickPower((id_list[j] - id) % q, -1, q)) % q
    lambda_i =lambda_i %q
    w_prime=(k_i-e_prime*lambda_i*si)%q
    #print('lamda_i:',lambda_i)
    return r_prime,w_prime,lambda_i

def DTCH_verify(hk,h,m_prime,r_prime):
    m_prime_hash = m_prime + str(r_prime[0])
    #e = H1(m_hash)
    e_prime = H1(m_prime_hash)
    #if (r[0]*quickPower(hk,e,p)*quickPower(g,r[1],p))%p==(r_prime[0]*quickPower(hk,e_prime,p)*quickPower(g,r_prime[1],p))%p:
    if h==((r_prime[0]*quickPower(hk,e_prime,p)*quickPower(g,r_prime[1],p))%p):
        print('Verify Success!')
    else:
        print('Verify False!')
if __name__ == "__main__":
    print('calculating...')
    print('')

    length = 256# 随机大素数长度
    result = primeFactorization(length)
    g = getGenerator(result)
    m = 'i sent first message'  # 消息1
    m_prime = 'second message'
    #print('p:',p)
    # 假设有t个参与者，每个参与者有一个唯一的ID

    participants = list(range(1, t + 1))
    # 每个参与者的ID
    id_list = [random.randint(1, 100) for _ in range(t)]

    #测试Time of KGen
    ts_KGen=time.perf_counter()
    # 使用线程池并行化每个参与者的计算过程
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(participant_PreparePoly, id_list) for participant_id in participants]
        results = [future.result() for future in concurrent.futures.as_completed(futures)]
    #print('(C_i,f_ij) from all Participants:',results)
    C=[results[i][0] for i in range(t)]
    #print('C:',C)
    F_ji=[]
    for i in range(t):
        F_ji.append([])
        for k in range(t):
            F_ji[i].append(results[k][1][i])
    #print('f_ji for every i:',F_ji)
    #使用线程池并行化每个参与者的承诺验证过程，每个参与者返回（si,pki,pk）
    with concurrent.futures.ThreadPoolExecutor() as executor:
        verification_futures = [executor.submit(participant_check,id,f_ji,C) for
                              id, f_ji in zip(id_list,F_ji)]
        verification_results = [future.result() for future in concurrent.futures.as_completed(verification_futures)]
    #print('(s_i,pk_i,pk):',verification_results)
    te_KGen=time.perf_counter()

    hk=verification_results[0][2]
    #print('系统公钥为：',hk)
    #打印公私秘钥份额，系统公钥
    for i in range(t):
        print('参与者'+str(id_list[i])+'的最终私钥份额(si,pk_i):', (verification_results[i][0],verification_results[i][1]))
    print('time of KGen(ms):',(te_KGen-ts_KGen)*1000)

    #准备Forge算法的形参sk_shares并按顺序传递
    sk_shares=[]
    for i in range(t):
        sk_shares.append(verification_results[i][0])
    #print('sk_shares:',sk_shares)
    #计算Time of Hash
    ts_Hash=time.perf_counter()
    h,r=DCH_Hash(hk,m)
    #print('h,(r,w):',h,r)
    te_Hash = time.perf_counter()
    print('time of Hash(ms):',(te_Hash-ts_Hash)*1000)

    #测试Forge
    ts_Forge = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor() as executor:
        Forge_PrePare_futures = [executor.submit(participants_forge_Prepare,id) for id in id_list]
        Forge_PrePare_results = [future.result() for future in concurrent.futures.as_completed(Forge_PrePare_futures)]

    #print('Forge_PrePare_results:',Forge_PrePare_results)

    k_i=[Forge_PrePare_results[i][0] for i in range(t)]
    K=1
    for i in range(t):
        K*=Forge_PrePare_results[i][1]%p
    K=K%p

    with concurrent.futures.ThreadPoolExecutor() as executor:
        Forge_futures = [executor.submit(participants_forge,id,K,k_i,si,m_prime,h) for id,k_i,si in zip(id_list,k_i,sk_shares)]
        Forge_results = [future.result() for future in concurrent.futures.as_completed(Forge_futures)]
    r0=Forge_results[0][0]
    w_prime=0
    for i in range(t):
        w_prime+=Forge_results[i][1]
    w_prime=w_prime%q
    r_prime=(r0,w_prime)
    te_Forge = time.perf_counter()
    print('time of Forge(ms):',(te_Forge-ts_Forge)*1000)

    lamda=[Forge_results[i][2] for i in range(t)]
    s=0
    c=1
    for i in range(t):
        s+=(lamda[i]*sk_shares[i])
        c*=C[i][0]
    s=s%q
    c=c%p
    print(c==quickPower(g,s,p))

    #测试Verify
    ts_Verify=time.perf_counter()
    DTCH_verify(hk,h,m_prime,r_prime)
    te_Verify=time.perf_counter()
    print('time of Verify(ms):',(te_Verify-ts_Verify)*1000)