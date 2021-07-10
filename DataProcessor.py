from os import makedirs
import os.path as path

import numpy as np   
import pandas as pd  

from scipy.spatial import distance

import string
import spacy

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from argparse import ArgumentParser

import DataCollector as dc


import time

# Globals
data_dir    = ""
img_dir   = ""
dst_text_features = ""

"""
Parsing the arguments:
* directory with data
* directory with images
* directory with text features
*     dst_image_features = "/home/gojourney/Uni/SS21/DS/DsPJ2021-Data/image_features"

Example usage: 
python3 DataProcessor.py -d "/home/gojourney/SS2021/DS/DsPJ2021-Data/data" -i "/home/gojourney/SS2021/DS/DsPJ2021-Data/images" -tf "/home/gojourney/SS2021/DS/DsPJ2021-Data/text_features"
"""
def parse_arguments():
    global data_dir
    global img_dir
    global dst_text_features
    global dst_image_features

    parser = ArgumentParser(description='Parse arguments')
    parser.add_argument('-d', '--data_dir', help='path to data directory',
                        required=True)
    parser.add_argument('-i', '--img_dir', 
                        help='path to images directory',
                        required=True)
    parser.add_argument('-tf', '--dst_text_features', 
                        help='path to directory to save text features',
                        required=True)
    parser.add_argument('-ti', '--dst_image_features', 
                        help='path to directory to save image features',
                        required=True)

    args = parser.parse_args()
    data_dir = args.data_dir
    img_dir = args.img_dir
    dst_text_features = args.dst_text_features
    dst_image_features = args.dst_image_features

def custom_tokenizer(text : str, remove_last : bool = True):

    punctuations = string.punctuation

    nlp = spacy.load("de_core_news_sm")

    nlp_text = nlp(text.replace("\xad", ""))
    nlp_text = [token.lemma_.lower().strip() for token in nlp_text if (not token.is_stop) and (token.lemma_.lower() not in punctuations)]
    nlp_text = [token.replace("\xad", "") for token in nlp_text]

    if remove_last:
        nlp_text = nlp_text[:-1]

    return nlp_text

def fit_bow(corpus : pd.DataFrame, test_corpus : pd.DataFrame = None, max_features : int = None):

    bow_vectorizer = CountVectorizer(tokenizer = custom_tokenizer, ngram_range = (1,1), max_features = max_features)
    bow_train_corpus = bow_vectorizer.fit_transform(corpus)

    if test_corpus is not None:

        bow_test_corpus = bow_vectorizer.transform(test_corpus)
        return bow_vectorizer, bow_train_corpus, bow_test_corpus

    return bow_vectorizer, bow_train_corpus

def fit_tfidf(corpus : pd.DataFrame, test_corpus : pd.DataFrame = None, max_features : int = None):

    tfidf_vectorizer = TfidfVectorizer(tokenizer = custom_tokenizer, max_features = max_features)
    tfidf_train_corpus = tfidf_vectorizer.fit_transform(corpus)

    if test_corpus is not None:

        tfidf_test_corpus = tfidf_vectorizer.transform(test_corpus) 
        return tfidf_vectorizer, tfidf_train_corpus, tfidf_test_corpus  

    return tfidf_vectorizer, tfidf_train_corpus 

def fit_doc2vec(corpus : pd.DataFrame, test_corpus : pd.DataFrame = None, 
                epochs : int = 100, vector_size : int = 256, alpha : float = 0.025, min_count : int = 1):

    corpus = [custom_tokenizer(doc) for doc in corpus.values]
    tagged_corpus = [TaggedDocument(words=doc, tags=[i]) for i, doc in enumerate(corpus)]

    model_d2v = Doc2Vec(vector_size = vector_size, alpha = alpha, min_count = min_count)
    model_d2v.build_vocab(tagged_corpus)

    for epoch in range(epochs):
        model_d2v.train(tagged_corpus, total_examples = model_d2v.corpus_count, epochs = model_d2v.epochs)

    doc2vec_train_corpus = np.zeros((len(corpus), vector_size))
    for i in range(len(corpus)):
        doc2vec_train_corpus[i] = model_d2v.dv[i]

    if test_corpus is not None:

        test_corpus = [custom_tokenizer(doc) for doc in test_corpus.values]

        doc2vec_test_corpus = np.zeros((len(test_corpus), vector_size))
        for i in range(len(test_corpus)):
            doc2vec_test_corpus[i] = model_d2v.infer_vector(test_corpus[i])

        return model_d2v, doc2vec_train_corpus, doc2vec_test_corpus

    return model_d2v, doc2vec_train_corpus

def calculate_most_similar(vectors : pd.DataFrame, target_vectors : pd.DataFrame, index_vector : pd.DataFrame = None):

    distances = distance.cdist(np.array(list(vectors.values)), np.array(list(target_vectors.values)), "cosine")
    min_index = np.argmin(distances, axis = 1)
    max_similarities = [1 - distances[i][min_index[i]] for i in range(0, len(min_index))] 

    if index_vector is not None:
        min_index = [index_vector[midx] for midx in min_index]

    return min_index, max_similarities


# Image Processing Functions:

from tqdm import tqdm

import tensorflow as tf
import tensorflow_datasets as tfds

from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import load_img

from tensorflow.keras.applications import VGG19
from tensorflow.keras.applications import ResNet152V2
from tensorflow.keras.applications import DenseNet201

def load_images(idata, IMG_SIZE = (224, 224)):

    paths = idata["ipath"].values
    images = [np.array(load_img(path, target_size=IMG_SIZE)) for path in paths]

    return np.array(images)

def transform_cnn(images, pretrained_model = DenseNet201, include_top = True, layer = None, batch_size = None): 

    model = pretrained_model(include_top = include_top, weights = 'imagenet')

    if layer is not None:
        
        intermediate_layer_model = Model(inputs=model.input,
                                         outputs=model.get_layer(layer).output)

        output = intermediate_layer_model.predict(np.array(images), batch_size = batch_size)

    else:
        output = model.predict(np.array(images), batch_size = batch_size)

    return output


def save_results(results : list, dst : str):

    with open(dst, 'wb') as f:

        for result in results:

            try:
                result = result if type(result) == np.ndarray else result.toarray()
            except:
                result = np.array(result)

            np.save(f, result)

def load_results(dst : str):

    results = []

    with open(dst, 'rb') as f:

        try:
            while (True):
                results.append(np.load(f))
        except:
            return results

    return results

def load_results_append(results, dst : str):
    results = list(results[0])
    with open(dst, 'rb') as f:

        try:
            while (True):
                results = np.append(results,np.load(f), axis=0)
        except:
            return results
    
    return results


        
    


if __name__ == "__main__":
    parse_arguments()

    data, adata, idata = dc.merge_train_data([path.join(data_dir, "01.tsv"), path.join(data_dir, "02.tsv")],
                                             [path.join(img_dir, "01"), path.join(img_dir, "02")])

    data_text, data_img = dc.get_test_data(path.join(data_dir, "03_data.tsv"), path.join(data_dir, "03_label.tsv"), path.join(img_dir, "03"))

    [dst_file_bow, dst_file_tfidf, dst_file_doc2vec] = [path.join(dst_text_features, 'bow_data.npy'), 
                                                        path.join(dst_text_features, 'tfidf_data.npy'),
                                                        path.join(dst_text_features, 'doc2vec_data.npy')]

    
    # Generate Image-Features (VGG19, ResNet152, DenseNet201) for 'Training'-data:
    
    [dst_file_vgg_fc1] = [path.join(dst_image_features, 'vgg_fc1_data.npy')]

    print("Start: Load Images.")
    images = load_images(idata)
    print("End: Load Images.")

    vgg_fc1_features = transform_cnn(images.copy(), pretrained_model = VGG19, include_top = True, layer = "fc1") 
    save_results([vgg_fc1_features], dst_file_vgg_fc1) 
    print("VGG19 - 1")

    """
    vgg_fc2_features = transform_cnn(images.copy(), pretrained_model = VGG19, include_top = True, layer = "fc2")  
    save_results([vgg_fc2_features], dst_file_vgg_fc2)
    print("VGG19 - 2")

    resnet_features = transform_cnn(images.copy(), pretrained_model = ResNet152V2, include_top = True, layer = "avg_pool")
    save_results([resnet_features], dst_file_resnet)
    print("ResNet")

    densenet_features = transform_cnn(images.copy(), pretrained_model = DenseNet201, include_top = True, layer = "avg_pool") 
    save_results([densenet_features], dst_file_densenet)
    print("DenseNet") 
    """
    

    # Generate Text-Features (BOW, TFIDF, DOC2VEC) for 'Training' and 'Test'-data:
       
    # Set Text-Feature Parameters:
    vector_size, epochs = 4096, 100

    _, train_bow, test_bow = fit_bow(adata["text"], test_corpus = data_text["text"], max_features = vector_size)
    save_results([train_bow, test_bow], dst_file_bow)

    _, train_tfidf, test_tfidf = fit_tfidf(adata["text"], test_corpus = data_text["text"], max_features = vector_size)
    save_results([train_tfidf, test_tfidf], dst_file_tfidf)

    _, train_doc2vec, test_doc2vec = fit_doc2vec(adata["text"], test_corpus = data_text["text"], epochs = epochs, vector_size = vector_size)
    save_results([train_doc2vec, test_doc2vec], dst_file_doc2vec)
    

    # Compute the most similar article (text) from the 'Training'-articles for each 'Test'-article:
    """
    result_bow, result_tfidf, result_doc2vec = load_results(dst_file_bow), load_results(dst_file_tfidf), load_results(dst_file_doc2vec)

    data_text, data_img = dc.get_test_data(path.join(data_dir, "03_data.tsv"), path.join(data_dir, "03_label.tsv"), path.join(img_dir, "03"))

    adata["bow"], adata["tfidf"], adata["doc2vec"] = list(result_bow[0]), list(result_tfidf[0]), list(result_doc2vec[0])
    data_text["bow"], data_text["tfidf"], data_text["doc2vec"] = list(result_bow[1]), list(result_tfidf[1]), list(result_doc2vec[1])

    bow_most_similar, bow_cosine_similarity = calculate_most_similar(data_text["bow"], adata["bow"], adata["aid"])
    tfidf_most_similar, tfidf_cosine_similarity = calculate_most_similar(data_text["tfidf"], adata["tfidf"], adata["aid"])
    doc2vec_most_similar, doc2vec_cosine_similarity = calculate_most_similar(data_text["doc2vec"], adata["doc2vec"], adata["aid"])

    data_text["bow_most_similar"], data_text["bow_cosine_similarity"] = bow_most_similar, bow_cosine_similarity
    data_text["tfidf_most_similar"], data_text["tfidf_cosine_similarity"] = tfidf_most_similar, tfidf_cosine_similarity
    data_text["doc2vec_most_similar"], data_text["doc2vec_cosine_similarity"] = doc2vec_most_similar, doc2vec_cosine_similarity

    del data_text["bow"]
    del data_text["tfidf"]
    del data_text["doc2vec"]
    data_text.to_csv(path.join(data_dir, "03_data_processed.tsv"), sep='\t')   
    """