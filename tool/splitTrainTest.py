import sys
import random

try:
    datapath = sys.argv[1]
    num_pos_test = int(sys.argv[2])
    num_neg_test = int(sys.argv[3])
    motif_posit_path = sys.argv[4]
except IndexError:
    print('python splitTrainTest [data] [num_pos_test] [num_neg_test] [motif_position_list]')
    exit(0)

def fileToList(fpath):
    posList = list()
    negList = list()
    with open(fpath) as fpr:
        fpr.readline()
        for line in fpr:
            if line[0] == '+':
                posList.append(line)
            else:
                negList.append(line)
        
    return posList, negList


posList, negList = fileToList(datapath)
with open(motif_posit_path) as fpr:
    motif_posit_list = [line for line in fpr if len(line)>1]

#shuffle pos samples
tmp = zip(posList, motif_posit_list)
random.shuffle(tmp)
posList, motif_posit_list = zip(*tmp)
#shuffle neg samples
random.shuffle( negList )

numPosTrain = len(posList) - num_pos_test
numPosTest = num_pos_test
numNegTrain = len(negList) - num_neg_test
numNegTest = num_neg_test

with open(datapath+'.train','w') as fpw:
    fpw.write( str(numPosTrain+numNegTrain) + '\n')
    for i in range(numPosTrain):
        fpw.write(posList[i])
    for i in range(numNegTrain):
        fpw.write(negList[i])

with open(motif_posit_path+'.train','w') as fpw:
    for i in range(numPosTrain):
        fpw.write( motif_posit_list[i] )
    
with open(datapath+'.test','w') as fpw:
    fpw.write( str(numPosTest+numNegTest) + '\n')
    for i in range(numPosTrain, numPosTrain+numPosTest):
        fpw.write(posList[i])
    for i in range(numNegTrain, numNegTrain+numNegTest):
        fpw.write(negList[i])

with open(motif_posit_path+'.test','w') as fpw:
    for i in range(numPosTrain, numPosTrain+numPosTest):
        fpw.write( motif_posit_list[i] )
