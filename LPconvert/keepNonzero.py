import sys
import io

try:
    inPath = sys.argv[1]
    outPath = sys.argv[2]
except IndexError:
    print('python keepNonzero.py [input file] [output file]')
    sys.exit(0)

with open(inPath) as fpr:
    with open(outPath,'w') as fpw:
        for line in fpr:
            if float(line.split()[1]) > 1e-2 :
                fpw.write(line)
