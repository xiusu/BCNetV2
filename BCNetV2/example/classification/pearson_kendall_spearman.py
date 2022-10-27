
"""
Pearson Rho, Spearman Rho, and Kendall Tau

Correlation algorithms

Drew J. Nase

Expects path to a file containing data series - 
one per line, separated by one or more spaces.
"""

import math
import sys
import string
import json
from itertools import combinations

# if len(sys.argv) > 1:
#     #data file given as arg
#     filename = sys.argv[1]
# else:
#     sys.exit("Usage: python " + sys.argv[0] + " [matrix filename]")

# x = []
# y = []

# def split_values(v):
#  buff = map(string.strip, string.split(v, " "))
#  x.append(float(buff[0]))
#  y.append(float(buff[1]))

#x, y must be one-dimensional arrays of the same length

#Pearson algorithm
def pearson(x, y):
    assert len(x) == len(y)
    q = lambda n: len(n) * sum(map(lambda i: i ** 2, n)) - (sum(n) ** 2)
    return (len(x) * sum(map(lambda a: a[0] * a[1], zip(x, y))) - sum(x) * sum(y)) / math.sqrt(q(x) * q(y))

#Spearman algorithm
def spearman(x, y):
    assert len(x) == len(y)
    q = lambda n: map(lambda val: sorted(n).index(val) + 1, n)
    d = sum(map(lambda x, y: (x - y) ** 2, q(x), q(y)))
    return 1.0 - 6.0 * d / float(len(x) * (len(y) ** 2 - 1.0))

#Kendall algorithm
def kendall(x, y):
    assert len(x) == len(y)
    c = 0 #concordant count
    d = 0 #discordant count
    t = 0 #tied count
    for (i, j) in combinations(range(len(x)), 2):
        s = (x[i] - x[j]) * (y[i] - y[j])
        if s:
            c += 1
            d += 1
            if s > 0:
                t += 1
            elif s < 0:
                t -= 1
        else:
            if x[i] - x[j]:
                c += 1
            elif y[i] - y[j]:
                d += 1
    return t / math.sqrt(c * d)




def sub2score(json_file):
    temp_json = {}
    for i in json_file["Result"]:
        # print(f'subnet: {i["subnet"]}, score: {i["score"]}')
        temp_json[str(i["subnet"])] = i["score"]

    return temp_json


def Top_k_list(Result_dict, My_dict, K = 100):

    Re_num_score = {}
    Re_k_list = []
    My_k_list = []
    for num in Result_dict:
        if int(num) < 4444444:
            Re_num_score[num] = Result_dict[num]["mean"]
    sorted_subnet = sorted(Re_num_score.items(), key=lambda i: i[1], reverse=True)
    sorted_subnet_key = [x[0] for x in sorted_subnet]
    Result_topk = sorted_subnet_key[:K]
    for i in Result_topk:
        Re_k_list.append(Re_num_score[i])
        My_k_list.append(My_dict[i])
    
    print(f'key: {Result_topk}, Re_k_list: {Re_k_list}, My_k_list: {My_k_list}')

    return Re_k_list, My_k_list










My_results = "json_results/bin20_min1_bcnet_3_resnet.json"
Record_results = "json_results/record_json/Results_ResNet.json"


with open(My_results,'r') as load_f:
    My_dict = json.load(load_f)

with open(Record_results,'r') as load_f:
    Re_dict = json.load(load_f)

My_temp = sub2score(My_dict)



# Re_list, My_list = Top_k_list(Re_dict, My_temp, K = 100)


My_list = []
Re_list = []

for num in Re_dict:
    if int(num) < 4444444:
        if '1' not in num:
            Re_list.append(Re_dict[num]["mean"])
            My_list.append(My_temp[num])


print('Pearson Rho: %f' % pearson(My_list, Re_list))

print('Spearman Rho: %f' % spearman(My_list, Re_list))

print('Kendall Tau: %f' % kendall(My_list, Re_list))