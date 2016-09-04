import sys
import io

try:
    data_fpath = sys.argv[1]
    clamp_sample = int(sys.argv[2])
    clamp_h = int(sys.argv[3])
except IndexError:
    print('python clamp.py [data] [clamp sample id] [clamp h_value]')
    sys.exit(0)

with open(data_fpath+'.c'+str(clamp_sample)+'_'+str(clamp_h),'w') as fpw:
    with open(data_fpath) as fpr:
        line = fpr.readline()
        fpw.write(line)
        for i, line in enumerate(fpr):
            if i==clamp_sample:
                tokens = line.split(', ')
                fpw.write(tokens[0]+', ')
                fpw.write(tokens[1].split(' . ')[clamp_h])
                fpw.write(' . \n')
            else:
                fpw.write(line)
