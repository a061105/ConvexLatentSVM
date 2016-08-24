import sys
import io
import random

def randSentence(sen_len, voc_size):
    sen = list()
    for i in range(sen_len):
        sen.append( str(random.randrange(voc_size)) )
    return sen

def randPos(length):
    return random.randrange(length)

def add_noise(key_sen, num_noise_char, voc_size):
    T = len(key_sen)
    key_sen_noise = list(key_sen)
    for i in range(num_noise_char):
        pos = randPos(T)
        key_sen_noise[pos] = str(random.randrange(voc_size))
    return key_sen_noise


############# Script Begins Here ################

try:
    num_pos_doc = int(sys.argv[1])
    num_neg_doc = int(sys.argv[2])
    voc_size = int(sys.argv[3])
    doc_len = int(sys.argv[4])
    sen_len = int(sys.argv[5])
    num_noise_char = int(sys.argv[6])
except IndexError:
    print('python3 randDoc [num_pos_doc] [num_neg_doc] [voc_size] [doc_len] [sen_len] [#noisy_char]');
    exit(0)

num_doc = num_pos_doc + num_neg_doc

key_sen = randSentence(sen_len, voc_size)
with open('keysentence','w') as fp:
    fp.write(' '.join(key_sen))

#random sentences
data = list()
for i in range(num_doc):
    doc = list()
    for j in range(doc_len):
        sen = randSentence(sen_len, voc_size)
        doc.append(sen)
    data.append(doc)

#motif sentence
key_pos = [None]*num_pos_doc;
for i in range(0,num_pos_doc):
    key_pos[i] = randPos(len(doc))
    data[i][ key_pos[i] ] = add_noise(key_sen, num_noise_char, voc_size)
    #data[i][ key_pos[i] ] = key_sen

#write key positions
with open('keypos','w') as fpw:
    for p in key_pos:
        fpw.write(str(p)+'\n')

#write to file
with open('docs','w') as fp:
    fp.write(str(len(data))+'\n')
    for i in range(0,num_doc):
        if i < num_pos_doc:
            fp.write('+1, ')
        else:
            fp.write('-1, ')
        fp.write(' . '.join( \
                (' '.join(sen)) for sen in data[i] \
                ) )
        fp.write('\n')
