# use weapon
#https://blog.csdn.net/your_answer/article/details/79131000


F_Matric = []
F_Constrain = 100000

Cin = 0.72 #(percent的平方关系)

#Build F_matric
for i in range(self.model.net):
    if isinstance(self.model.net, conv)
        temp = Cin * Cout * H* W * k * k / stride / stride
        if net.group == Cin == Cout:
            F_C = F_C - temp / Cout
            F_Matric.append([0] * len(list_percent))

        elif net.group == 1:
            for j in range(len(list_percent)):
                F_Matric.append(temp * list_percent[j])

        else:
            raise RuntimeError("Not support group type, group is {}, cin in is {}, cout is {}".format( net.group, Cin, Cout))

    elif isinstance(self.model.net, linear):
        temp = Cin * Cout
        for j in range(len(list_percent)):
            F_Matric.append(temp * list_percent[j])


# pop last layer
for _ in range(len(list_percent)):
    F_Matric.pop(-1)

# skip line merge
for i in skip_list:
    temp1 = i.pop(0)
    for j in i:
        for k in range(len(list_percent)):
            F_Matric[temp1 * len(list_percent) + k] += F_Matric[j * len(list_percent) + k]
            F_Matric[j * len(list_percent) + k] = 0


# del 0
#this place need del all 0 in F_Matric

# build linear uneq constrint
A_ub = F_Matric
B_ub = [F_C]

#Build linear function
A_eq = []
B_eq = []
for i in range(0, len(F_Matric), len(list_percent)):
    temp = [0] * len(F_Matric)
    for j in range(len(list_percent)):
        temp[i * len(list_percent) + j] = 1

    A_eq.append(temp)
    B_eq.append(1.0)


# build bound
bounds = []
for i in range(len(F_Matric)):
    bounds.append([0,1])

for i in range(len(bounds)):
    bounds[i] = tuple(bounds[i])
bounds = tuple(bounds)


x3=(0,7)
# S is the important factor
res=op.linprog(S,A_ub,B_ub,A_eq,B_eq,bounds=bounds)

print("搜索结果是 {}".format(res.fun))
print("最佳概率分布是 {}".format(res.x))

# rearrange the list into 2D list and make it suitable for layer wise calculation
temp_list = []
for i in range(0, len(res.x), len(list_percent)):
    temp2_list = []
    for j in range(len(list_percent)):
        temp2_list.append(res.x[i * len(list_percent) + j])

    temp_list.append(temp2_list)

# restore the list into the model lenghth list
temp_list = channel_fangda(temp_list, skip_list)

#check eq
assert temp_list = number_layers









