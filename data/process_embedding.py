import numpy as np
import pickle as pkl
import os
from tqdm import tqdm


# SETTINGS >>>



dic_dir = 'processed/mpqg_substitute_a_vocab_include_a/'
dic_name = 'vocab.dic'
embedding_name = 'glove840b_vocab300.npy'

data_dir = '../../data/'
glove = 'glove.840B.300d'



# SETTINGS <<<



# LOAD & PROCESS GloVe >>>

print("here")

if not os.path.exists('processed/'+ glove + '.dic.npy'):
    # Load GloVe
    f = open(data_dir + glove + '.txt')
    lines = f.readlines()
    f.close()

    # Process GloVe
    embedding = dict()
    for line in tqdm(lines):
        splited = line.split()
        embedding[splited[0]] = [float(s) for s in splited[1:]]

    # Save processed GloVe as dic file
    #np.save('processed/' + glove + '.dic', embedding)
else:
    print("here")
    embedding = np.load('processed/' + glove + '.dic.npy').item()





# LOAD & PROCESS GloVe <<<


# PRODUCE PRE-TRAINED EMBEDDING >>>


# Load vocabulary

print("here")

with open(os.path.join(dic_dir, dic_name),"rb") as f:
    vocab = pkl.load(f)

print("here")


# Initialize random embedding and extract pre-trained embedding

embedding_vocab =  np.random.ranf((len(vocab), 300)) -  np.random.ranf((len(vocab), 300))

embedding_vocab[0] = 0.0 # vocab['<PAD>'] = 0
embedding_vocab[1] = embedding['<s>'] # vocab['<GO>'] = 1
embedding_vocab[2] = embedding['EOS'] # vocab['<EOS>'] = 2
embedding_vocab[3] = embedding['UNKNOWN'] # vocab['<UNK>'] = 3

print("here")

unk_num = 0
for word, idx in tqdm(vocab.items()):
    if word in embedding:
        embedding_vocab[idx] = embedding[word]
    else:
        unk_num += 1

np.save(os.path.join(dic_dir, embedding_name), embedding_vocab)



# PRODUCE PRE-TRAINED EMBEDDING <<<
