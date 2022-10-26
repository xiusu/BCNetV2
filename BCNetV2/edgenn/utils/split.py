import random

dic = {}
with open('train.txt', 'r') as f:
    for line in f:
        line = line.strip()
        (s, i) = line.split(' ')
        i = int(i)
        if i not in dic:
            dic[i] = []
        dic[i].append(s)

print([(i, len(dic[i])) for i in dic.keys()])

train = []
val = []
for i, v in dic.items():
    v_val = random.sample(v, 50)
    for vv in v_val:
        v.remove(vv)
    v_train = v
    for vv in v_train:
        train.append(f'{vv} {i}')
    for vv in v_val:
        val.append(f'{vv} {i}')

print(f'{random.sample(train, 5)}  {len(train)}')
print(f'{random.sample(val, 5)} {len(val)}')

with open('nas_train.txt', 'w') as f:
    for v in train:
        f.write(f'{v}\n')
with open('nas_val.txt', 'w') as f:
    for v in val:
        f.write(f'{v}\n')
