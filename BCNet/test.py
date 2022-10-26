import numpy as np
import torch.nn as nn
import random
import math
import copy
import torch.nn.functional as F
import torch
import matplotlib.pyplot as plt
from torch.autograd import Variable


def Sampler_1Dto2D(sample_list, Length):
    # Length must from the "def Model_length" and be the full size of model
    sample_flist = []
    L1 = 0
    L2 = 0
    Len = len(Length)
    for i in range(Len):
        L2 = L2 + Length[i]
        sample_flist.append(sample_list[L1:L2])
        L1 = L2
    #assert len(sample_list) == L1, f'skip_line must have more than 2 layers'
    return sample_flist

def Channel_wise_dropout(bin_size_list):
    Total_Percent = random.uniform(0.5, 0.6)
    print(Total_Percent)
    Total_list = []
    Total_channels = [12,14,15]
    assert len(Total_channels) == len(bin_size_list), f'model_channels do not equal to bin_size_list.'
    bin_number_list = [j // i for i, j in zip(bin_size_list, Total_channels)]
    print("bin_number_list is {}".format(bin_number_list))
    for i in range(len(bin_size_list)):
        Channel_list = []
        oup = bin_number_list.pop(0)
        Save_oup = int(math.ceil(oup * Total_Percent))
        test_list = [0] * oup
        for i in range(Save_oup):
            test_list[i] = 1
        random.shuffle(test_list)
        # test_list = np.random.permutation(test_list)

        T_channels = Total_channels.pop(0)
        gama = bin_size_list.pop(0)
        assert int(T_channels - oup * gama) == 0, f'T_channels divide oups must be Integer.'
        for i in test_list:
            List_temp = [i for _ in range(gama)]
            Channel_list.append(List_temp)
        Channel_list = sum(Channel_list, [])
        Total_list.append(Channel_list)
    return Total_list

def list_change(oup):
    List = []
    i = 0
    for x in oup:
        if x == 1:
            List.append(i)
        elif x != 0:
            raise RuntimeError('x in list must be 0 or 1')
        i += 1
    return List

def Sampler_deduction(deduction_loc, sample_list):
    # only support 2D extension_loc list
    deduction_loc_copy = copy.deepcopy(deduction_loc)
    deduction_loc_copy.sort(reverse=True)
    for i in deduction_loc_copy:
        del i[0]
        for j in i:
            sample_list[j] = 0
            #del sample_list[j]
    Len =  len(sample_list)

    j = 0
    for i in range(Len):
        if sample_list[i-j] == 0:
            del sample_list[i-j]
            j = j + 1
    return sample_list


def Sampler_extension(extension_loc, sample_list):
    #only support 2D extension_loc list
    for i in extension_loc:
        if len(i) == 0:
            return sample_list

    extension_loc_copy = copy.deepcopy(extension_loc)
    extension_loc_copy.sort(reverse = False)

    temp = []
    for i in extension_loc_copy:
        del i[0]
        for j in i:
            temp.append(j)

    temp.sort()
    extension_loc_copy = copy.deepcopy(extension_loc)
    #print("temp is {}, sample_list is {}, extension_loc is {}".format(temp, sample_list,extension_loc_copy))
    for i in temp:
        sample_list.insert(i, 0)
        assert i <= len(sample_list), 'deduction_loc index exceed sample_list length, sample_list is {}, extension_loc is {}'.format(sample_list,extension_loc_copy)

    for i in extension_loc_copy:
        first = i.pop(0)
        first_list = sample_list[first]
        for j in i:
            sample_list[j] = first_list
    return sample_list

def FLOPs_aware(FLOPs_list):
    # produce channel FLOPs_aware bin_size_list
    #how to change max_bin to half after each genetic algorithm
    min_list = min(FLOPs_list)
    max_list = max(FLOPs_list)
    FLOPs_aware_list = [ math.ceil(math.log(1 / i * max_list, 2)) for i in FLOPs_list]
    #FLOPs_aware_list = [ math.ceil(math.log(i / min_list, 2)) for i in FLOPs_list]
    FLOPs_aware_list = [ 1 * math.pow(2, i) for i in FLOPs_aware_list]
    bin_size_list = [min(i, 32 * 1) for i in FLOPs_aware_list]
    #bin_size_list = np.clip(FLOPs_aware_list, a_max = self.max_bin * self.min_bin)
    return bin_size_list

def test(a, **kwargs):
    print(a)
    for k,w in kwargs.items():
        print(k)

def BinaryRandomProbabilitySampling(size, Prob = 0.5):
    low = Prob - 0.5
    high = Prob + 0.5
    val = np.random.uniform(low= low, high = high, size =size)
    return (val > 0.5).astype(np.int)

class Net():
    def __init__(self,b):

        self.a= Net_temp(b=b)

class Net_temp():
    def __init__(self, **kwargs):
        for k in kwargs:
            setattr(self, k, kwargs[k])
        if isinstance(self.a, dict):
            print(1)

#'''
def Init_sample(Record_Efficient, P_train, F_Constraint, skip_list):
    # 等会儿要按照每层的概率来乘起来

    skip_list_copy = copy.deepcopy(skip_list)
    FLOPs_list = [884736, 294912, 524288, 1572864, 884736, 2359296, 3538944, 1327104, 3538944, 3538944,
              1327104, 4718592, 6291456, 1769472, 6291456, 6291456, 1769472, 6291456, 6291456, 442368,
              3145728, 6291456, 884736, 6291456, 6291456, 884736, 6291456, 6291456, 884736, 6291456,
              6291456, 884736, 9437184, 14155776, 1327104, 14155776, 14155776, 1327104, 14155776,
              14155776, 331776, 5898240, 9830400, 552960, 9830400, 9830400, 552960, 9830400, 9830400,
              552960, 19660800, 26214400, 12800]

    skip_list_copy = copy.deepcopy(skip_list)
    # skip line merge
    for i in skip_list_copy:
        if len(i) == 0:
            break
        temp1 = i.pop(0)
        for j in i:
            FLOPs_list[temp1] += FLOPs_list[j]
            FLOPs_list[j] = 0

    skip_list_copy = copy.deepcopy(skip_list)
    for i in skip_list_copy:
        if len(i) == 0:
            break
        temp1 = i.pop(0)
        for j in i:
            for k in range(len(P_train)):
                Record_Efficient[temp1][k] += Record_Efficient[j][k]
                Record_Efficient[j][k] = 0

    # pop last fc layer, output layer out_channel don't change, but input_channel influence
    # FLOPs_list.pop(-1)

    # merge skip line index
    j = 0
    for i in range(len(FLOPs_list)):
        if FLOPs_list[i - j] == 0:
            del FLOPs_list[i - j]
            del Record_Efficient[i - j]
            j = j + 1

    # real layer num
    Layer_Num = len(FLOPs_list)  # L
    P_train_Num = len(P_train)  # N

    # construct M metric
    M = []
    for i in range(1, Layer_Num - 1):
        for j in range(P_train_Num):
            temp = [0] * ((Layer_Num - 1) * P_train_Num)
            for k in range(i * P_train_Num, (i + 1) * P_train_Num):
                temp[k] = FLOPs_list[i] * P_train[j]

            M.append(temp)

    # append last few layers 0
    for j in range(P_train_Num):
        temp = [0] * ((Layer_Num - 1) * P_train_Num)
        M.append(temp)

    print("M is {}".format(M))

    M = torch.FloatTensor(M)

    # layer init
    search_Num = 1000
    Index = 0.01
    arf = 1  # 0.9999
    beta = 1
    while (1):
        search_Num = search_Num - 1
        # max search Num
        if (search_Num <= 0):
            break
        X_ = torch.randn(1, (Layer_Num - 1) * P_train_Num, requires_grad=True)
        gradient = torch.ones_like(X_) * 0.01

        for i in range(1000):
            if i % 100 == 0:
                print("now is {}".format(i))

            X = torch.exp(X_)
            # softmax for every layer, this place wrong
            Y_ = torch.ones_like((X))
            for i in range(Layer_Num - 1):
                temp = torch.sum(X[0, i * P_train_Num: (i + 1) * P_train_Num])
                for j in range(P_train_Num):
                    Y_[0, i * P_train_Num + j] = X[0, i * P_train_Num + j] / temp

            # X mul prob
            X_P = torch.ones_like(X)
            for i in range(Layer_Num - 1):
                for j in range(P_train_Num):
                    X_P[0, i * P_train_Num + j] = Y_[0, i * P_train_Num + j] * P_train[j]

            # first Quadratic term + first_layer & last_layer
            temp_constraint = X_P.mm(M).mm(X_P.t()) + torch.sum(FLOPs_list[0] * X_P[0, :P_train_Num]) + torch.sum(
                FLOPs_list[-1] * X_P[0, (Layer_Num - 2) * P_train_Num:])

            # object function
            object_constrain = torch.tensor([0.0])
            for i in range(Layer_Num - 1):
                for j in range(P_train_Num):
                    object_constrain += Y_[0, i * P_train_Num + j] * Record_Efficient[i][j]

            Y = object_constrain + Index * pow(temp_constraint/F_Constraint - 1, 2)#torch.clamp(temp_constraint - F_Constraint, min=0)
            Z = torch.ones(1, (Layer_Num - 1) * P_train_Num) * Y
            Z.backward(gradient)
            gradient *= arf
            Index *= beta

            X_ = X_.data - X_.grad.data
            X_ = torch.tensor(X_,
                              requires_grad=True)  # torch.randn(1, (Layer_Num - 1) * P_train_Num, requires_grad=True)

        print("object_constrain is {}， temp_constraint is {}, F_Constraint is {}, clamp value is {}".format(object_constrain, temp_constraint, F_Constraint, pow(temp_constraint/F_Constraint - 1, 2)))

        if pow(temp_constraint/F_Constraint - 1, 2) > 0.001:
            Index = Index * 1.5

        #elif (temp_constraint - F_Constraint) < -F_Constraint/10:
        #   Index = Index / 1.5
        #    Flag_down = True


        else:
            print("good Index is {}".format(Index))
            break
        print("Index value is {}".format(Index))

    Y_ = Y_.tolist()
    P_list = []
    for i in range(Layer_Num - 1):
        temp = []
        for j in range(P_train_Num):
            temp.append(Y_[0][i * P_train_Num + j])
        P_list.append(temp)

    # skip_list_copy = copy.deepcopy(skip_list)
    # P_list = Sampler_extension(skip_list_copy, P_list)
    print("best P_list is {},temp_constraint is {}, clamp value is {}".format(P_list, temp_constraint, torch.clamp(temp_constraint - F_Constraint, min=0)))
    return P_list



def Neq_Random(P_list, Num):
    P_list_copy_ = copy.deepcopy(P_list)

    Len = len(P_list_copy_[0])
    P_list_copy = copy.deepcopy(P_list)
    for i in range(len(P_list_copy_)):
        for j in range(Len):
            P_list_copy[i][j] = sum(P_list_copy_[i][:(j + 1)])

    Index_list = []
    for i in range(Num):
        temp_list = []
        for j in range(len(P_list_copy)):
            temp = np.random.uniform(low=0, high=P_list_copy[j][Len-1])
            Index = 0
            while (1):
                if (P_list_copy[j][Index] >= temp):
                    temp_list.append(Index)
                    break
                Index += 1
        Index_list.append(temp_list)

    #Prob_list = []
    #for i in range(len(Index_list)):
    #    Prob_list.append(P_train[Index_list[i]])
    return Index_list



'''
skip_list = [[]]
FLop_list = [100, 100, 400, 400]
P_train = [0.25, 1]
Record_Efficient = [[2, 1], [2, 1], [2,1], [2, 1]]
F_Constraint = 800
P_list = Init_sample(Record_Efficient, P_train, F_Constraint, skip_list)
print(P_list)
'''

'''
skip_list = [[3,5], [4,6]]
FLop_list = [213., 124, 331, 412, 222, 123, 999, 398]
Record_Efficient = [[12,32,11,20], [9, 21,22,13], [31,21,22,41], [21.,21,22,30], [123,213,211,21], [213., 412, 222, 121], [213,222,111,211], [91,212,333,412]]
P_train = [0.25, 0.50, 0.75, 1.0]
Model_Channels = [6,8,12,16,30,20,10]

F_Constraint = 1400.
P_list = Init_sample(Record_Efficient, P_train, F_Constraint, skip_list)

Index_list = Neq_Random(P_list, 3)

#Index_list = np.array(Index_list)

size = (3, 7)
Random_M = BinaryRandomProbabilitySampling(size, Prob = 0.5)
Random_L = list(Random_M)

for i in range(len(Random_L)):
    Index_list.append(Random_L[i])

Index_list = np.array(Index_list)

Temp = [0.25,0.5, 0.75, 1]
ub = np.ones(6) + len(Temp) -1
print(ub)
'''

# 500 epoch test
a = [0,1,2,3,4,6]
print(a.pop(2))
print(a)
'''
#500epoch
Record_Efficient = [[0.12436026228074626, 0.12433595921275634, 0.12430497601562876, 0.12438767322258193],
                    [0.12436026228074626, 0.12433595921275634, 0.12430497601562876, 0.12438767322258193],
                    [0.12436242559194528, 0.12437063628659419, 0.12432755460948561, 0.12432475517095644],
                    [0.1243425267261972, 0.12436773943859603, 0.12438502712603536, 0.12429412235476864],
                    [0.1243425267261972, 0.12436773943859603, 0.12438502712603536, 0.12429412235476864],
                    [0.1243249084644129, 0.12433243338237238, 0.12437842056333641, 0.12434776382288588],
                    [0.1244006923606535, 0.12430090400643232, 0.1243483854230042, 0.12434039426443615],
                    [0.1244006923606535, 0.12430090400643232, 0.1243483854230042, 0.12434039426443615],
                    [0.1243249084644129, 0.12433243338237238, 0.12437842056333641, 0.12434776382288588],
                    [0.12430322352354653, 0.12432789370327328, 0.1243923736012771, 0.12436509072419251],
                    [0.12430322352354653, 0.12432789370327328, 0.1243923736012771, 0.12436509072419251],
                    [0.12432564899159332, 0.12436026935980762, 0.1243710639138768, 0.12432835441207078],
                    [0.1243159296539267, 0.12438084137832377, 0.12433158701240082, 0.12435701483175832],
                    [0.1243159296539267, 0.12438084137832377, 0.12433158701240082, 0.12435701483175832],
                    [0.12432564899159332, 0.12436026935980762, 0.1243710639138768, 0.12432835441207078],
                    [0.12441817439648269, 0.12437522749825088, 0.12431397372068885, 0.12429017659353722],
                    [0.12441817439648269, 0.12437522749825088, 0.12431397372068885, 0.12429017659353722],
                    [0.12432564899159332, 0.12436026935980762,0.1243710639138768, 0.12432835441207078],
                    [0.12430727759332301, 0.12438299226320687, 0.12433299385094403, 0.12436375004708701],
                    [0.12430727759332301, 0.12438299226320687, 0.12433299385094403, 0.12436375004708701],
                    [0.12434104686742933, 0.12434447136686136, 0.12441559852802993, 0.1242920027558285],
                    [0.12432733317873533, 0.12434816947532194, 0.12433662021732217, 0.1243710732553104],
                    [0.12432733317873533, 0.12434816947532194, 0.12433662021732217, 0.1243710732553104],
                    [0.12434104686742933, 0.12434447136686136, 0.12441559852802993, 0.1242920027558285],
                    [0.12434767491996765, 0.12438466464638456, 0.12432254744130836, 0.1243313376829084],
                    [0.12434767491996765, 0.12438466464638456, 0.12432254744130836, 0.1243313376829084],
                    [0.12434104686742933, 0.12434447136686136, 0.12441559852802993, 0.1242920027558285],
                    [0.12436512257606994, 0.12435926189320758, 0.12432621994183862, 0.12433222545204456],
                    [0.12436512257606994, 0.12435926189320758, 0.12432621994183862, 0.12433222545204456],
                    [0.12434104686742933, 0.12434447136686136, 0.12441559852802993, 0.1242920027558285],
                    [0.12438250814776562, 0.12437164581531888, 0.12433421781646965, 0.12430216348634952],
                    [0.12438250814776562, 0.12437164581531888, 0.12433421781646965, 0.12430216348634952],
                    [0.12433355498924684, 0.12436368561830062, 0.12435304706857604, 0.12433326615687225],
                    [0.12433115453311168, 0.12436857699134131, 0.12438442643753977, 0.12430382575066207],
                    [0.12433115453311168, 0.12436857699134131, 0.12438442643753977, 0.12430382575066207],
                    [0.12433355498924684, 0.12436368561830062, 0.12435304706857604, 0.12433326615687225],
                    [0.12438062025366457, 0.12434229614701009, 0.12432739590292019, 0.12433399348361975],
                    [0.12438062025366457, 0.12434229614701009, 0.12432739590292019, 0.12433399348361975],
                    [0.12433355498924684, 0.12436368561830062, 0.12435304706857604, 0.12433326615687225],
                    [0.12436067763103939, 0.12431520937000289, 0.12437330488234608, 0.12433534208879449],
                    [0.12436067763103939, 0.12431520937000289, 0.12437330488234608, 0.12433534208879449],
                    [0.12437321829285429, 0.12430689681196061, 0.12437861714706995, 0.12432981791774748],
                    [0.12437156606240585, 0.12433385430080503, 0.1243283731872587, 0.12435121381291515],
                    [0.12437156606240585, 0.12433385430080503, 0.1243283731872587, 0.12435121381291515],
                    [0.12437321829285429, 0.12430689681196061, 0.12437861714706995, 0.12432981791774748],
                    [0.1243458126790646, 0.12434543388428129, 0.12433723068341904, 0.12435385635839859],
                    [0.1243458126790646, 0.12434543388428129, 0.12433723068341904, 0.12435385635839859],
                    [0.12437321829285429, 0.12430689681196061, 0.12437861714706995, 0.12432981791774748],
                    [0.12441203987331663, 0.12431746793278052, 0.12431066297846112, 0.12435138333977752],
                    [0.12441203987331663, 0.12431746793278052, 0.12431066297846112, 0.12435138333977752],
                    [0.12431862319131651, 0.1243654667241897, 0.12430840582923403, 0.12439815318975715],
                    [0.12429424538322972, 0.12435504792038637, 0.12435358300186378, 0.12438716350857455]]

P_train = [0.25, 0.5, 0.75, 1.0]
                #222354816
F_Constraint = 150354816.0
                #296473088
#14526566912
skip_list = [[0, 1], [3, 4], [5, 8], [6, 7], [9, 10], [11, 14, 17], [12, 13], [15, 16], [18, 19], [20, 23, 26, 29], [21, 22], [24, 25], [27, 28], [30, 31], [32, 35, 38], [33, 34], [36, 37], [39, 40], [41, 44, 47], [42, 43], [45, 46], [48, 49]]
FLOPs_list = [884736, 294912, 524288, 1572864, 884736, 2359296, 3538944, 1327104, 3538944, 3538944,
              1327104, 4718592, 6291456, 1769472, 6291456, 6291456, 1769472, 6291456, 6291456, 442368,
              3145728, 6291456, 884736, 6291456, 6291456, 884736, 6291456, 6291456, 884736, 6291456,
              6291456, 884736, 9437184, 14155776, 1327104, 14155776, 14155776, 1327104, 14155776,
              14155776, 331776, 5898240, 9830400, 552960, 9830400, 9830400, 552960, 9830400, 9830400,
              552960, 19660800, 26214400, 12800]

print(sum(FLOPs_list))
P_list = Init_sample(Record_Efficient, P_train, F_Constraint, skip_list)
B_Prob_list = Neq_Random(P_list, Num = 5)
print("best pro list is {}".format(B_Prob_list))


#L_Channels, R_Channels = Sample_channels(Prob_list, Model_Channels)
'''

