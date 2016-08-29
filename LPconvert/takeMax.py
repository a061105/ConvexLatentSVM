import io
import sys

try:
    fpath = sys.argv[1]
    pre = sys.argv[2]
except IndexError:
    print('python takeMax.py [fpath] [pre_str] (post_str)')
    exit(0)

try:
    post = sys.argv[3]
except IndexError:
    post = None


max_val = -1e300
with open(fpath) as fpr:
    for line in fpr:
        
        if (pre not in line):
            continue

        if post == None:
            val = float(line.split(pre)[1])
        else:
            val = float(line.split(post)[0])

        if val > max_val:
            max_val = val
    

print('max='+str(max_val))
