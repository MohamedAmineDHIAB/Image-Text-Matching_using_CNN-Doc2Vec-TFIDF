#python script to infer document vectors from pretrained doc2vec model on english wikipedia dump
import gensim.models as g
import codecs
import pandas as pd
import numpy as np
from tqdm import tqdm

#parameters
model="data/doc2vec.bin"



#inference hyper-parameters
start_alpha=0.01
infer_epoch=1000

#load model
m = g.Doc2Vec.load(model)

df=pd.read_csv('data/batch_01_02_03_deepl_titles.tsv', sep='\t',encoding='utf-8')
texts_list=df.loc[:,'deepl_title'].tolist()
tokenized_texts=[x.strip().split() for x in texts_list]

#infer title vectors

n=len(tokenized_texts)
vecs=np.zeros((300,n))
print('number of documents : ',n,'\n')
print('--------------------------------')
print('shape of vecs :', vecs.shape,'\n')
for i in tqdm(range(n)):
    vecs[:,i]=m.infer_vector(tokenized_texts[i], alpha=start_alpha, steps=infer_epoch)

#save embedding vectors of english titles in numpy file
output_file="data/batch_01_02_03_titles_vectors.npy"
with open(output_file,'wb') as f :
    np.save(f, vecs)
