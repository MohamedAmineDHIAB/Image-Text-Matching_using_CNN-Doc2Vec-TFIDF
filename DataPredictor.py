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

def calculate_n_most_similar(vectors, target_vectors, n_similar = 10):

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

    [dst_file_bow, dst_file_tfidf, dst_file_doc2vec] = [path.join(dst_text_features, 'bow_data.npy'), 
                                                        path.join(dst_text_features, 'tfidf_data.npy'),
                                                        path.join(dst_text_features, 'doc2vec_data.npy')]

    data_text, data_img = dc.get_test_data(path.join(data_dir, "03_data.tsv"), path.join(data_dir, "03_label.tsv"), path.join(img_dir, "03"))
    """
    # Make random predictions:
    random_pred = random_baseline(data_text, data_img)
    print(random_pred)
    """


    #Calculate similarity baseline based on tfidf/doc2vec
    data, adata, idata = dc.merge_train_data([path.join(data_dir, "01.tsv"), path.join(data_dir, "02.tsv"), path.join(data_dir, "03.tsv")],
                                             [path.join(img_dir, "01"), path.join(img_dir, "02"), path.join(img_dir, "03")])
    
    result_tfidf = dp.load_results(dst_file_tfidf)
    
    adata["tfidf"] = list(result_tfidf[0])
    data_text["tfidf"] = list(result_tfidf[1])

    tfidf_most_similar, tfidf_cosine_similarity = dp.calculate_most_similar(data_text["tfidf"], adata["tfidf"], adata["aid"])
    data_text["tfidf_most_similar"], data_text["tfidf_cosine_similarity"] = tfidf_most_similar, tfidf_cosine_similarity

    iv_data = dp.load_results_append(dp.load_results(dst_file_img01), dst_file_img02)
    test_iv_data = dp.load_results(dst_file_img03)

    #Make prediction based on cosine similarity
    cosine_pred = cosine_similarity_baseline(data_text, data, iv_data, test_iv_data)
    save_baseline_results(cosine_pred, path, 'cosine_results.csv', 'csv')
    

    #result_tfidf = load_results(path.join(path,'eng_titles_vectors.npy'))
    """
    train = pd.read_csv(path.join(path,'train.csv'))
    dev = pd.read_csv(path.join(path, 'dev.csv'))
    test = pd.read_csv(path.join(path, 'test.csv'))
    train_length = len(train) + len(dev)
    test_length = len(test)

    train_data = adata.iloc[0:train_length]
    test_data = adata.iloc[train_length:]
    train_data["tfidf"] = list(result_tfidf[0])[:train_length] #result_tfidf[0].T
    test_data["tfidf"] = list(result_tfidf[0])[train_length:] #result_tfidf[0].T

    tfidf_most_similar, tfidf_cosine_similarity = calculate_most_similar(test_data["tfidf"], train_data["tfidf"], train_data["aid"])
    test_data["tfidf_most_similar"], test_data["tfidf_cosine_similarity"] = tfidf_most_similar, tfidf_cosine_similarity
    
    dt = train.append(dev, ignore_index=True, sort=False)
    itrain = (iv_data)[:train_length]
    itest = (iv_data)[train_length:]
    cosine_pred = cosine_similarity_baseline(test_data, dt, itrain, [itest], test, 100)
    save_baseline_results(cosine_pred, path, 'cosine_results.csv', 'csv')
    """



