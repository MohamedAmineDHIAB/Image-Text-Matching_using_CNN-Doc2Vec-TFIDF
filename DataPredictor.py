from os import makedirs
import os.path as path

import numpy as np   
import pandas as pd  

import spacy
from scipy.spatial import distance

import DataCollector as dc
import DataProcessor as dp

"""
Parsing the arguments:
* directory with data
* directory with images
* directory with text features
* directory with image features

Example usage: 
python3 DataPredictor.py -d "data_dir" -i "img_dir" -tf "dst_text_features" -ti "dst_img_features"
"""
def parse_arguments():
    global data_dir
    global img_dir
    global dst_text_features
    global dst_img_features

    parser = ArgumentParser(description='Parse arguments')
    parser.add_argument('-d', '--data_dir', help='path to data directory',
                        required=True)
    parser.add_argument('-i', '--img_dir', 
                        help='path to images directory',
                        required=True)
    parser.add_argument('-tf', '--dst_text_features', 
                        help='path to text features directory',
                        required=True)
    parser.add_argument('-ti', '--dst_img_features', 
                        help='path to images features directory',
                        required=True)

    args = parser.parse_args()
    data_dir = args.data_dir
    img_dir = args.img_dir
    dst_text_features = args.dst_text_features
    dst_img_features = args.dst_img_features

def random_baseline(adata : pd.DataFrame, idata: pd.DataFrame):

    adata["random_baseline"] = idata["iid"].sample(n=adata.shape[0]).values
    return adata

def calculate_n_most_similar(vectors, target_vectors, n_similar = 100):

    distances = distance.cdist(np.array(list(vectors)), np.array(list(target_vectors)), "cosine")
    d = distances[0]
    n_mins = sorted(range(len(d)), key = lambda sub: d[sub])[:n_similar]

    return n_mins

def cosine_similarity_baseline(adata, data, iv_data, test_iv_data, data_img, n_similar = 10):

    n_most_similarslist = list()

    for index, row in adata.iterrows():
        idx = data.index[data['aid'] == row["tfidf_most_similar"]].tolist()
        train_img = iv_data[idx[0]]
        n_mins = calculate_n_most_similar([train_img], test_iv_data[0], n_similar)
        for i in range(len(n_mins)):
            n_mins[i] = data_img.iloc[n_mins[i]]['iid'] 
        n_most_similarslist.append(n_mins)
    
    adata["similarity_baseline"] = n_most_similarslist

    return adata

def save_baseline_results(pred, path, filename, ext = 'csv'):
    if (ext == 'csv'):
        results = pred[['aid', pred.columns[1]]]
        results.to_csv(path.join(path,filename), index=False)
    else:
        f = open(path.join(path,filename), "w")
        for line in pred[pred.columns[1]]:
            for item in line:
                f.write(str(item) + "\t")
            f.write("\n")
        f.close()

if __name__ == "__main__":
    
    [dst_file_img01, dst_file_img02, dst_file_img03] = [path.join(dst_text_features, 'vgg_fc1_data.npy'), 
                                                        path.join(dst_text_features, 'vgg_fc1_data2.npy'),
                                                        path.join(dst_text_features, 'vgg_fc1_data3.npy')]


    dst_file_tfidf = path.join(dst_text_features, 'tfidf_data.npy')

    data_text, data_img = dc.get_test_data(path.join(data_dir, "MediaEvalNewsImagesBatch03articles.tsv"), path.join(data_dir, "MediaEvalNewsImagesBatch03images.tsv"), path.join(img_dir, "img-2019-03"))

    #Calculate similarity baseline based on tfidf
    data, adata, idata = dc.merge_train_data([path.join(data_dir, "MediaEvalNewsImagesBatch01.tsv"), path.join(data_dir, "MediaEvalNewsImagesBatch02.tsv")],
                                             [path.join(img_dir, "img-2019-01"), path.join(img_dir, "img-2019-02")])
    
    result_tfidf = dp.load_results(dst_file_tfidf)
    
    adata["tfidf"] = list(result_tfidf[0])
    data_text["tfidf"] = list(result_tfidf[1])

    tfidf_most_similar, tfidf_cosine_similarity = dp.calculate_most_similar(data_text["tfidf"], adata["tfidf"], adata["aid"])
    data_text["tfidf_most_similar"], data_text["tfidf_cosine_similarity"] = tfidf_most_similar, tfidf_cosine_similarity

    iv_data = dp.load_results_append(dp.load_results(dst_file_img01), dst_file_img02)
    test_iv_data = dp.load_results(dst_file_img03)

    #Make prediction based on cosine similarity
    cosine_pred = cosine_similarity_baseline(data_text, data, iv_data, test_iv_data, data_img, "tfidf_most_similar")
    save_baseline_results(cosine_pred, path, 'cosine_results_de_100.csv', 'csv')


    adata["doc2vec"] = list(dp.load_results(path.join(data_dir,"doc2vec_batch01_02_titles_vectors.npy"))[0].T)
    k = dp.load_results(path.join(dst_text_features, "doc2vec_batch03_titles_vectors.npy"))[0].T
    data_text["doc2vec"] = list(k)

    d2v_most_similar, d2v_cosine_similarity = calculate_n_most_similar(data_text["doc2vec"], adata["doc2vec"], adata["aid"])
    data_text["doc2vec_most_similar"], data_text["doc2vec_cosine_similarity"] = d2v_most_similar, d2v_cosine_similarity

        #Make prediction based on cosine similarity
    cosine_pred = cosine_similarity_baseline(data_text, data, iv_data, test_iv_data, data_img, "doc2vec_cosine_similarity")
    save_baseline_results(cosine_pred, path, 'cosine_results_eng_100.csv', 'csv')





