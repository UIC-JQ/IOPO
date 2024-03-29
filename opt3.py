"""
use for build the formula
Helen 
"""
import numpy as np
import random
import math
import copy
import sys

def compute_local_eng_cost(task_size=None, time_threshold=None, user_compute_speed=None,
                           user_compute_power=None, penalty=None):
    """
    计算本地能量消耗 = 计算时间 * 单位时间消耗能量大小(i.e.功率) + penalty
        - 如果计算时间超过了time_threshold, 则penalty=penalty,
        - 否则penalty = 0
    """

    compute_time_required = task_size / user_compute_speed

    if compute_time_required > time_threshold:
        return compute_time_required * user_compute_power + penalty
    
    return compute_time_required * user_compute_power

def compute_upload_eng_cost(task_size=None, 
                            package_size=None,
                            time_threshold=None,
                            uav_compute_speed=None,
                            uav_compute_power=None,
                            penalty=None,
                            uav_coordinate=None,
                            user_coordinate=None,
                            user_transmit_power=None,
                            data_config=None
                            ):
    # 记录不同的energy cost
    compute_eng_cost = transmit_eng_cost = 0

    # 机器上的计算时间
    compute_time_required = task_size / uav_compute_speed
    # 机器上的计算能耗
    compute_eng_cost = compute_time_required * uav_compute_power

    # # 传输task到机器上需要的总时间
    trans_speed = __compute_user_to_uav_trans_speed(uav_coordinate, user_coordinate, data_config)
    transmit_time_required = package_size / trans_speed
    # 传输包到机器上需要的总能耗
    transmit_eng_cost = transmit_time_required * user_transmit_power

    total_time_required = compute_time_required + transmit_time_required

    if total_time_required > time_threshold:
        return (compute_eng_cost + transmit_eng_cost) + penalty
    
    return compute_eng_cost + transmit_eng_cost, compute_time_required, transmit_time_required

def __compute_user_to_uav_trans_speed(uav_coordinate, user_coordinate, data_config):
    uav_coordinate = np.array(uav_coordinate)
    user_coordinate = np.array(user_coordinate)
    
    l0 = np.array(data_config.IRS_l0_coordinate)

    c = data_config.SPEED_OF_LIGHT
    f = data_config.SUB_CHANNEL_FREQUENCY
    K = data_config.SUB_CHANNEL_K
    z = data_config.IRS_z_number
    x = data_config.IRS_x_number #IRS x-axis refector
    B = data_config.CHANNEL_BANDWIDTH
    p = data_config.CHANNEL_POWER #信道传输功率
    T = data_config.TIME_SLOT_LENGTH   #s equal to 0.125ms,time of one slot
    sigma2 = 10**((-174)/10 - 3)#-174 dBm/Hz Gaussian nosie power
    phi = np.zeros(x * z)
    delta = data_config.IRS_delta

    def _rate():
        g_hat_gain = _ghatGain(_g_gain)
        h_gain     = _hGain()

        res = B * np.math.log((1+(p*abs(g_hat_gain+h_gain)**2)/sigma2), 2)
        result = res/(1/T) #bit/s to bit/slot
        return result
    
    def _hGain():
        du = np.linalg.norm(uav_coordinate - user_coordinate)
        result = c/(4*np.pi*f*du)*np.exp(-1j*2*np.pi*f*du/c+(-K*du)/2)

        return result

    def _ghatGain(g_gain):
        a = np.matmul(_er(), _phi_diag())
        b = np.matmul(a, _eu())
        result = _g_gain() * b
        return result
    
    def _g_gain():
        dm = np.linalg.norm(l0 - uav_coordinate)
        dr = np.linalg.norm(l0 - user_coordinate)

        dp = dr + dm
        result = c/(8*np.sqrt(np.pi**3)*f*dp)*np.exp(-1j*2*np.pi*f*dp/c+(-K*dp)/2)

        return result

    def _er():
        tem = []
        tem2 = []
        theta = []
        e_r = []
        r0 = np.linalg.norm(uav_coordinate - l0)
        
        # 垂直板
        for j in range(1, z + 1):
            # 水平板子
            for k in range(1, x + 1):
                l_t = ((k-1)*delta,0,(j-1)*delta)
                tem.append(l_t)

        tem = np.array(tem)
        for j in tem:
            tem2.append(np.sum((l0-uav_coordinate)*j))#无人机m坐标乘距离

        for k in tem2:
            theta.append(2*np.pi*f*k/(c*r0))#无人机计算theta 

        for k in theta:
            e_r.append(np.exp(-1j*k))#e_r

        return e_r
    
    def _eu():
        tem = []
        tem2 = []
        theta = []
        tem = []
        e_u = []
        ru = np.linalg.norm(user_coordinate - l0)

        for j in range(1,z+1):
            for k in range(1,x+1):
                l_t=((k-1)*delta,0,(j-1)*delta)
                tem.append(l_t)

        tem = np.array(tem)
        for k in tem:
            tem2.append(np.sum((l0-user_coordinate)*k))#用户u坐标乘距离

        for k in tem2:
            theta.append(2*np.pi*f*k/(c*ru))#用户计算theta

        for k in theta:
            e_u.append(np.exp(-1j*k))#e
        return e_u
    
    def _phi_diag():
        temp = []
        for i in phi:
            temp.append(np.exp(1j*i))
        x = np.diag(temp)
        return x

    return _rate()

def whale(h, b, data_config, need_stats=False, optimize_phi=False, compute_local_eng_cost=False, compute_upload_eng_cost=False, PENALTY=None, remove_board=False, test_stage=False):
    # for equation 1 - 3
    '''
    IRS在x-z平面
    the location of the first reflecting element is l0(4,0,4)-任意a,0,c
    The IRS is located in the middle of X-axis with the hight of 4m,Nx=4,Nz=4
    ,and δx=δz=5mm
    '''
    
    if PENALTY is None:
        penalty = data_config.overtime_penalty #j for solution violate constrain 
    else:
        penalty = PENALTY

    c = data_config.SPEED_OF_LIGHT     #speed of the light m/s
    T = data_config.TIME_SLOT_LENGTH   #s equal to 0.125ms,time of one slot
    
    delta = data_config.IRS_delta
    user = data_config.user_number     #number of users
    uav = data_config.uav_number       #number of uavs

    z_u = data_config.user_computational_capacity #the computation rate of UEDs cycles/slot
    p_u = data_config.user_computation_power      #用户计算功率 j/slot
    p_t = data_config.user_transmit_power #用户传输功率 j/slot

    l0 = data_config.IRS_l0_coordinate#IRS第一块坐标

    z_m = data_config.uav_computational_capacity #the computation rate of UAVs cycles/slot
    p_m = data_config.uav_computational_power    #无人机功率 j/slot = 600W
    lm = data_config.uav_coordinate
    
    x = data_config.IRS_x_number #IRS x-axis refector
    z = data_config.IRS_z_number #IRS z-axis refector
    # modify during test
    # x = 20 #IRS x-axis refector
    # z = 10 #IRS z-axis refector
    
    #bandwith & sub-channel
    p = data_config.CHANNEL_POWER #信道传输功率
    f = data_config.SUB_CHANNEL_FREQUENCY  #子频道的中央频率 Hz
    K = data_config.SUB_CHANNEL_K #每个子频道中央频率的吸收参数 db/m
    B = data_config.CHANNEL_BANDWIDTH
    
    def split_info(h):
        task = []
        h = np.array(h) #transfer input to np array
        size = user * 3
        lu,task = np.split(h,[size])
        lu = lu.reshape(-1, 3)

        #拆分data size 和cpu cycles
        task = task.reshape(-1, 3)
        data_u=[]
        cycle_u =[]
        time_u = []
        for i in task:
            data_u = np.append(data_u,i[0])
            cycle_u = np.append(cycle_u,i[1])
            time_u = np.append(time_u,i[2])

        return lu,data_u,cycle_u,time_u

    lu,data_u,cycle_u,time_u= split_info(h)

    # for equation 4
    sigma2 = 10**((-174)/10 - 3)#-174 dBm/Hz Gaussian nosie power
    b = np.array(b)
    b1=np.where(b==1)[0]


    def offload():
        pair = []
        res = []
        for j in b1:
            pair.append(int(np.ceil((j+1)/uav)-1))#user u
            if (j+1)%uav ==0:
                pair.append(uav-1)
            else:
                pair.append((j+1)%uav-1)#UAV m
            res.append(pair)
            pair = []
        return res

    off = offload() #the offload alloction

    #count the workload of one UAV
    def workload():
        freq_dict = {}
        for sublist in off:
            key = sublist[1]
            if key not in freq_dict:
                freq_dict[key] = 0
            freq_dict[key] += 1
        return(freq_dict)

    loaddic = workload()
    
    def local():
        lu =[]
        if b1.size == 0:
            for i in range(user):
                lu.append(i)
        else:
            b0 = b.reshape(-1, uav)
            sums = np.sum(b0,axis=1)
            lu=np.where(sums==0)[0]
        return lu

    def ddp(u,m):
        dm = np.sqrt(np.sum(np.square(lm[m] - l0)))
        dr = np.sqrt(np.sum(np.square(lu[u] - l0)))
        dp = dr+dm
        return dp
    def ddu(u,m):
        du = np.sqrt(np.sum(np.square(lm[m] - lu[u])))
        return du

    def gGain(u,m):
        dp = ddp(u,m)
        result = c/(8*np.sqrt(np.pi**3)*f*dp)*np.exp(-1j*2*np.pi*f*dp/c+(-K*dp)/2)
        return result

    def hGain(u,m):
        du = ddu(u,m)
        result = c/(4*np.pi*f*du)*np.exp(-1j*2*np.pi*f*du/c+(-K*du)/2)
        return result

    def ghatGain(u,m,phi):
        a = np.matmul(er(m), phi_diag(phi))
        b = np.matmul(a,eu(u))
        #print("g-gain ",gGain(u,m))
        result = gGain(u,m)*b
        #print("ghat - IRS gain",result)
        return result

    def rate(u, m, phi, remove_board=None):
        gain = abs(ghatGain(u,m,phi)+hGain(u,m))
        assert remove_board is not None

        if not remove_board:
            res = B*np.math.log((1+(p*abs(ghatGain(u,m,phi)+hGain(u,m))**2)/sigma2),2)
        else:
            res = B*np.math.log((1+(p*abs(hGain(u,m))**2)/sigma2),2)

        result = res/(1/T)#bit/s to bit/slot
        return result

    def er(m):
        tem = []
        tem2 = []
        theta = []
        e_r = []
        r0 = np.sqrt(np.sum(np.square(lm[m] - l0)))
        for j in range(1,z+1):
            for k in range(1,x+1):
                l_t = ((k-1)*delta,0,(j-1)*delta)
                tem.append(l_t)
        tem = np.array(tem)
        for j in tem:
            tem2.append(np.sum((l0-lm[m])*j))#无人机m坐标乘距离

        for k in tem2:
            theta.append(2*np.pi*f*k/(c*r0))#无人机计算theta 

        for k in theta:
            e_r.append(np.exp(-1j*k))#e_r

        return e_r

    def eu(u):
        tem = []
        tem2 = []
        theta = []
        tem = []
        e_u = []
        ru = np.sqrt(np.sum(np.square(lu[u] - l0)))
        for j in range(1,z+1):
            for k in range(1,x+1):
                l_t=((k-1)*delta,0,(j-1)*delta)
                tem.append(l_t)
        tem = np.array(tem)
        for k in tem:
            tem2.append(np.sum((l0-lu[u])*k))#用户u坐标乘距离

        for k in tem2:
            theta.append(2*np.pi*f*k/(c*ru))#用户计算theta

        for k in theta:
            e_u.append(np.exp(-1j*k))#e
        return e_u
    
    def phi_diag(phi):
        temp = []
        for i in phi:
            temp.append(np.exp(1j*i))
        x = np.diag(temp)
        #print(x)
        return x

    def E_offload(phi, remove_board=remove_board):
        E_tran = 0
        E_comp = 0
        ovt_log = set()
        avg_trans_speed = 0
        avg_trans_eng_cost = 0


        for j in off:
            u = j[0] 
            m = j[1]
            trans_speed = rate(u,m,phi, remove_board=remove_board)
            time_t = data_u[u] / trans_speed
            # print("transition time",time_t)
            time_c = cycle_u[u] / (z_m[m]/loaddic[m])
            # print('compute_time:', time_c)
            tran_enery = (data_u[u] / trans_speed) * p_t[u]
            if (time_c + time_t) > time_u[u]:
                # print("Task must not exceed the deadline")
                # add penalty
                E_tran = E_tran + tran_enery + penalty #p_t为用户传输功率
                ovt_log.add(u)
            else:
                E_tran = E_tran + tran_enery

            E_comp = E_comp + cycle_u[u]/z_m[m]*p_m[m]

            avg_trans_speed += trans_speed
            avg_trans_eng_cost += tran_enery
        
        result = E_tran+E_comp

        return result, ovt_log, avg_trans_speed / user, avg_trans_eng_cost / user
    
    def E_local():
        res = local()
        # print("local allocation ",res)
        E_temp = 0
        ovt_log = set()

        for j in res:
            time = cycle_u[j]/z_u[j]
            if time > time_u[j]:
                E_temp = E_temp + (cycle_u[j]/z_u[j]) * p_u[j] + penalty
                ovt_log.add(j)
            else:
                E_temp = E_temp + (cycle_u[j]/z_u[j]) * p_u[j]

        return E_temp, ovt_log
        

    # whale algorithm starts here
    # - target func
    def prob(phi):
        r1, _, _, _ = E_offload(phi)
        r2, _ = E_local()

        return r1 + r2
    
    def overtime_stat(phi, remove_board=None, test_stage=False):
        assert remove_board is not None
        _, log1, trans_speed, trans_eng_cost = E_offload(phi, remove_board)
        _, log2 = E_local()
        if not test_stage:
            return list(log1.union(log2))
            
        return list(log1.union(log2)), trans_speed, trans_eng_cost
    
    # def delete_board_exp(phi, flag):
    #     eng_cost, __logs, avg_trans_speed, avg_trans_eng_cost = E_offload(phi, remove_board=flag)
    #     if len(__logs) > 0:
    #         eng_cost += len(__logs) * penalty

    #     return eng_cost, avg_trans_speed, avg_trans_eng_cost 

    # whale class
    class whale:
        def __init__(self, fitness, dim, minx, maxx, seed):
            self.rnd = random.Random(seed)
            self.position = [0.0 for i in range(dim)]

            for i in range(dim):
                self.position[i] = ((maxx - minx) * self.rnd.random() + minx)

            self.fitness = fitness(self.position) # curr fitness


    # whale optimization algorithm(WOA)
    def woa(fitness, max_iter, n, dim, minx, maxx):
        rnd = random.Random(0)

        # create n random whales
        whalePopulation = [whale(fitness, dim, minx, maxx, i) for i in range(n)]

        # compute the value of best_position and best_fitness in the whale Population
        Xbest = [0.0 for i in range(dim)]
        Fbest = sys.float_info.max

        for i in range(n): # check each whale
            if whalePopulation[i].fitness < Fbest:
                Fbest = whalePopulation[i].fitness
                Xbest = copy.copy(whalePopulation[i].position)

        # main loop of woa
        Iter = 0
        while Iter < max_iter:

            # linearly decreased from 2 to 0
            a = 2 * (1 - Iter / max_iter)
            a2 = -1 + Iter*((-1)/max_iter)

            for i in range(n):
                A = 2 * a * rnd.random() - a
                C = 2 * rnd.random()
                b = 1
                l = (a2-1)*rnd.random()+1
                p = rnd.random()

                D = [0.0 for i in range(dim)]
                D1 = [0.0 for i in range(dim)]
                Xnew = [0.0 for i in range(dim)]
                Xrand = [0.0 for i in range(dim)]
                if p < 0.5:
                    if abs(A) > 1:
                        for j in range(dim):
                            D[j] = abs(C * Xbest[j] - whalePopulation[i].position[j])
                            Xnew[j] = Xbest[j] - A * D[j]
                    else:
                        p = random.randint(0, n - 1)
                        while (p == i):
                            p = random.randint(0, n - 1)

                        Xrand = whalePopulation[p].position

                        for j in range(dim):
                            D[j] = abs(C * Xrand[j] - whalePopulation[i].position[j])
                            Xnew[j] = Xrand[j] - A * D[j]
                else:
                    for j in range(dim):
                        D1[j] = abs(Xbest[j] - whalePopulation[i].position[j])
                        Xnew[j] = D1[j] * math.exp(b * l) * math.cos(2 * math.pi * l) + Xbest[j]

                for j in range(dim):
                    whalePopulation[i].position[j] = Xnew[j]

            for i in range(n):
                # if Xnew < minx OR Xnew > maxx
                # then clip it
                for j in range(dim):
                    whalePopulation[i].position[j] = max(whalePopulation[i].position[j], minx)
                    whalePopulation[i].position[j] = min(whalePopulation[i].position[j], maxx)

                whalePopulation[i].fitness = fitness(whalePopulation[i].position)

                if (whalePopulation[i].fitness < Fbest):
                    Xbest = copy.copy(whalePopulation[i].position)
                    Fbest = whalePopulation[i].fitness

            Iter += 1
        # end-while
        # returning the best solution
        return Xbest
    # ----------------------------

    #Begin whale optimization algorithm for target function
    dim = x*z #反射板总个数
    fitness = prob

    num_whales = data_config.optimize_num_whales
    max_iter = data_config.optimize_max_iter


    if optimize_phi:
        best_position = woa(fitness, max_iter, num_whales, dim,0,2*np.pi)
    else:
        best_position = np.zeros(dim)

    err = fitness(best_position)
    
    if need_stats:
        return best_position, err, overtime_stat(best_position, remove_board=remove_board, test_stage=test_stage)

    return best_position, err