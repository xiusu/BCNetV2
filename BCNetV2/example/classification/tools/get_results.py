import sys
import os
import yaml

dirs = sys.argv[1:]


for d in dirs:
    paths = list(os.listdir(d))
    paths.sort(reverse=True)
    for p in paths:
        if 'pth.tar' in p and p != 'last.pth.tar':
            args = yaml.safe_load(open(os.path.join(d, 'args.yaml'), 'r'))
            metrics = open(os.path.join(d, 'metrics.csv'), 'r').read().split('\n')[1:]
            metrics = [[int(x.split(',')[0]), float(x.split(',')[3])] for x in metrics if x != '']
            metrics.sort(key=lambda x: x[1], reverse=True)
            if metrics[0][1] < 50:
                break
            print(d, p)
            print(metrics[0])
            #print(args)
            break

