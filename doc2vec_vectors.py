# python script to infer document vectors from pretrained doc2vec model on english wikipedia dump
import gensim.models as g

import pandas as pd
import numpy as np
from tqdm import tqdm
from argparse import ArgumentParser


def parse_arguments():
    global b12_eng
    global b3_eng
    global dst_text_features

    parser = ArgumentParser(description='Parse arguments')
    parser.add_argument('-b12',
                        '--b12_eng',
                        help='batch01 and batch02 english titles files',
                        required=True)
    parser.add_argument('-b3',
                        '--b3_eng',
                        help='batch03 english titles files',
                        required=True)
    parser.add_argument('-tf',
                        '--dst_text_features',
                        help='path to text features destination',
                        required=True)

    args = parser.parse_args()
    b12_eng = args.b12_eng
    b3_eng = args.b3_eng
    dst_text_features = args.dst_text_features


if __name__ == "__main__":
    # parameters
    model = "data/doc2vec.bin"

    # inference hyper-parameters
    start_alpha = 0.01
    infer_epoch = 1000

    # load model
    m = g.Doc2Vec.load(model)
    for file in [b12_eng, b3_eng]:

        df = pd.read_csv('data/batch_01_02_03_deepl_titles.tsv',
                         sep='\t',
                         encoding='utf-8')
        texts_list = df.loc[:, 'deepl_title'].tolist()
        tokenized_texts = [x.strip().split() for x in texts_list]

        # infer title vectors

        n = len(tokenized_texts)
        vecs = np.zeros((300, n))
        print('number of documents : ', n, '\n')
        print('--------------------------------')
        print('shape of vecs :', vecs.shape, '\n')
        for i in tqdm(range(n)):
            vecs[:, i] = m.infer_vector(tokenized_texts[i],
                                        alpha=start_alpha,
                                        steps=infer_epoch)

        # save embedding vectors of english titles in numpy file
        if file == b12_eng:

            output_file = dst_text_features + "doc2vec_batch01_02_titles_vectors.npy"
            with open(output_file, 'wb') as f:
                np.save(f, vecs)
        else:
            output_file = dst_text_features + "doc2vec_batch03_titles_vectors.npy"
            with open(output_file, 'wb') as f:
                np.save(f, vecs)
