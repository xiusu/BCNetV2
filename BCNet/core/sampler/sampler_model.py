import torch
import torch.nn as nn
import copy
from core.utils.flops import count_flops
from pymoo.model.sampling import Sampling
import numpy as np
from pymoo.util.normalization import denormalize
from os.path import join
from torch.autograd import Variable
from core.utils.flops import count_sample_flops, Model_channels

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


def rebuild_model(model, rebuild_model, sampler_list, subnet=None, **kwargs):
    # model and rebuild_model tiqu chulai ,weight + bias + bn.parameter
    #if subnet is None:
    #    subnet = [0] * len(model)
    #assert len(model) == len(rebuild_model), f'sampler_model.py model must equal to rebuild_model in len.'

    #print("model is {}, rebuild_model is {}".format(model, rebuild_model))


    Inp = [0,1,2]
    for ms, rs, in zip(model.net, rebuild_model.net):
        for m, r in zip(ms.modules(), rs.modules()):
            if isinstance(m, torch.nn.Conv2d):
                oup = sampler_list.pop(0)
                oup = list_change(oup)
                Size = m.weight.data[:, :, :, :].size()[1]
                Groups = m.groups
                if Size == 1:
                    r.weight.data = m.weight.data[oup, :, :, :]

                    if m.bias is not None:
                        r.bias.data = m.bias.data[oup, :, :, :]
                elif Groups != Size:
                    r.weight.data = m.weight.data[oup, :, :, :]
                    r.weight.data = r.weight.data[:, Inp, :, :]

                    if m.bias is not None:
                        r.weight.data = m.weight.data[oup, :, :, :]
                        r.weight.data = r.weight.data[:, Inp, :, :]
                else:
                    raise RuntimeError("Groups != Size, you need to check this, Groups is {}, Size is {}".format(Groups, Size))
                Inp = oup

            if isinstance(m, torch.nn.BatchNorm2d):
                r.weight.data = m.weight.data[oup]
                if m.bias is not None:
                    r.bias.data = m.bias.data[oup]


            if isinstance(m, torch.nn.Linear):
                if len(sampler_list) == 0:
                    return rebuild_model
                oup = sampler_list.pop(0)
                oup = list_change(oup)
                Size = m.weight.data[:, :, :, :].size()[1]
                Groups = m.groups
                if Size == 1:
                    r.weight.data = m.weight.data[oup, :, :, :]

                    if m.bias is not None:
                        r.bias.data = m.bias.data[oup, :, :, :]
                elif Groups != Size:
                    r.weight.data = m.weight.data[oup, :, :, :]
                    r.weight.data = r.weight.data[:, Inp, :, :]

                    if m.bias is not None:
                        r.weight.data = m.weight.data[oup, :, :, :]
                        r.weight.data = r.weight.data[:, Inp, :, :]
                else:
                    raise RuntimeError("Groups != Size, you need to check this, Groups is {}, Size is {}".format(Groups, Size))
                Inp = oup

    return rebuild_model

def Init_sample(model, Record_Efficient, P_train, F_Constraint, skip_list, input_shape):
    # 等会儿要按照每层的概率来乘起来

    skip_list_copy = copy.deepcopy(skip_list)
    FLOPs_list = count_flops(model, FLOPs_list = True, input_shape = input_shape)
    Record_Efficient = list(Sampler_extension(skip_list_copy, Record_Efficient))

    #print("Record_Efficient is {}".format(Record_Efficient))
    #print("P_train is {}".format(P_train))
    #print("F_Constraint is {}".format(F_Constraint))
    #print("skip_list is {}".format(skip_list))
    #print("FLOPs_list is {}".format(FLOPs_list))

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
    Index = 0.5
    arf = 1  # 0.9999
    beta = 1
    while (1):
        search_Num = search_Num -1
        # max search Num
        if (search_Num <= 0):
            break
        X_ = torch.randn(1, (Layer_Num - 1) * P_train_Num, requires_grad=True)
        gradient = torch.ones_like(X_) * 0.01

        for i in range(1000):
            if i% 100 == 0:
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
            temp_constraint = X_P.mm(M).mm(X_P.t()) + torch.sum(FLOPs_list[0] * X_P[0,:P_train_Num]) + torch.sum(FLOPs_list[-1] * X_P[0,(Layer_Num - 2) * P_train_Num : ])

            # object function
            object_constrain = torch.tensor([0.0])
            for i in range(Layer_Num - 1):
                for j in range(P_train_Num):
                    object_constrain += Y_[0,i * P_train_Num + j] * Record_Efficient[i][j]

            Y = object_constrain + Index * pow(temp_constraint/F_Constraint - 1, 2)
            Z = torch.ones(1, (Layer_Num - 1) * P_train_Num) * Y
            Z.backward(gradient)
            gradient *= arf
            Index *= beta

            X_ = X_.data - X_.grad.data
            X_ = torch.tensor(X_, requires_grad=True)  #torch.randn(1, (Layer_Num - 1) * P_train_Num, requires_grad=True)

        print("object_constrain is {}， temp_constraint is {}, F_Constraint is {}, clamp value is {}".format(object_constrain, temp_constraint, F_Constraint, pow(temp_constraint/F_Constraint - 1, 2)))

        if pow(temp_constraint / F_Constraint - 1, 2) > 0.01:
            Index = Index * 1.3

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

    #skip_list_copy = copy.deepcopy(skip_list)
    #P_list = Sampler_extension(skip_list_copy, P_list)
    print("best P_list is {},temp_constraint is {}, clamp value is {}".format(P_list, temp_constraint, torch.clamp(temp_constraint - F_Constraint, min=0)))
    return P_list


def Neq_Random(P_list, P_train, skip_list, Model_Channels, input_shape, model, FLOPs_Constrain, Num):
    P_list_copy_ = copy.deepcopy(P_list)

    Len = len(P_list_copy_[0])
    P_list_copy = copy.deepcopy(P_list)
    for i in range(len(P_list_copy_)):
        for j in range(Len):
            P_list_copy[i][j] = sum(P_list_copy_[i][:(j + 1)])

    Index_list = []

    N = 0
    while (N  < Num):
        temp_list = []
        for j in range(len(P_list_copy)):
            temp = np.random.uniform(low=0, high=P_list_copy[j][Len-1])
            Index = 0
            while (1):
                if (P_list_copy[j][Index] >= temp):
                    temp_list.append(Index)
                    break
                Index += 1
        L_Channels, _ = Channel_list(P_train, skip_list, Model_Channels, temp_list)

        flops = count_sample_flops(model, L_Channels, input_shape=input_shape)
        print("flops is {}, FLOPs_constraint is {}".format(flops, FLOPs_Constrain))
        if (1 - flops/FLOPs_Constrain < 0.1) and (flops < FLOPs_Constrain):
            print("flops is {}, FLOPs_constraint is {}, divid is {}".format(flops, FLOPs_Constrain, flops/FLOPs_Constrain))
            Index_list.append(temp_list)
            N += 1

    assert len(Index_list) == Num, "Index_list len not eq to Num, len is {}, Num is {}, Index_list is {}".format(len(Index_list), Num, Index_list)

    #Prob_list = []
    #for i in range(len(Index_list)):
    #    Prob_list.append(P_train[Index_list[i]])
    return Index_list




def Channel_list(P_train, skip_list, Model_Channels, subnet_list):
    Prob_list = []
    for i in range(len(subnet_list)):
        Prob_list.append(P_train[subnet_list[i]])

    assert len(Prob_list) == len(Model_Channels), "Prob_list len must eq to Model_Channels, Prob_list is {}, Model_Channels is {}， Pro is {}, M_C is {}".format(len(Prob_list), len(self.Model_Channels), Prob_list, Model_Channels)

    L_Channels, R_Channels = Sample_channels(Prob_list, Model_Channels)

    L_Channels = Sampler_extension(skip_list, L_Channels)
    R_Channels = Sampler_extension(skip_list, R_Channels)

    return L_Channels, R_Channels


def Sample_channels(Prob_list, Model_Channels):
    L_Channels = []
    for i in range(len(Model_Channels)):
        L_Channels.append([0] * Model_Channels[i])

    Ac_Channels = [int(i * j) for i, j in zip(Prob_list, Model_Channels)]

    for i in range(len(L_Channels)):
        for j in range(Ac_Channels[i]):
            L_Channels[i][j] = 1

    R_Channels = []
    for i in range(len(L_Channels)):
        temp = copy.deepcopy(L_Channels[i])
        temp.reverse()
        R_Channels.append(temp)
    return L_Channels, R_Channels






def Model_length(model, subnet=None):
    Length = 0
    if subnet is None:
        subnet = [0] * len(model)
    for ms, idx in zip(model, subnet):
        for m in ms[idx].modules():
            if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear):
                Length += 1
    return Length

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

def Sampler_deduction(deduction_loc, sample_list):
    # only support 2D extension_loc list
    for i in deduction_loc:
        if len(i) == 0:
            return sample_list

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

def Sampler_2Dto1D(sample_list):
    return sum(sample_list, [])

def Sampler_1Dto2D(sample_list, Length):
    # Length must from the "def Model_length" and be the full size of model
    sample_flist = []
    L1 = 0
    while len(Length)>0:
        L2 = Length.pop(0)
        sample_flist.append(sample_list[L1:L2])
        L1 = L2 + 1

    assert len(sample_list) == L1, f'sample_list length must equal to the model length, sample_list is {sample_list}, Length is {L1}'
    return sample_flist


def BinaryRandomProbabilitySampling(size, Prob = 0.5):
    low = Prob - 0.5
    high = Prob + 0.5
    val = np.random.uniform(low= low, high = high, size =size)
    return (val < 0.5).astype(np.bool)
