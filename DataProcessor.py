from os import makedirs
import os.path as path

import numpy as np   
import pandas as pd  

from scipy.spatial import distance

import string
import spacy

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer


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
*     dst_image_features = "Image-Text-Matching_using_CNN-Doc2Vec/image_features"

Example usage: 
python3 DataProcessor.py -d "Image-Text-Matching_using_CNN-Doc2Vec/data" -i "Image-Text-Matching_using_CNN-Doc2Vec/image-cache" -tf "Image-Text-Matching_using_CNN-Doc2Vec/text_features" -ti "Image-Text-Matching_using_CNN-Doc2Vec/image_features" 
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


def fit_tfidf(corpus : pd.DataFrame, test_corpus : pd.DataFrame = None, max_features : int = None):

    tfidf_vectorizer = TfidfVectorizer(tokenizer = custom_tokenizer, max_features = max_features)
    tfidf_train_corpus = tfidf_vectorizer.fit_transform(corpus)

    if test_corpus is not None:

        tfidf_test_corpus = tfidf_vectorizer.transform(test_corpus) 
        return tfidf_vectorizer, tfidf_train_corpus, tfidf_test_corpus  

    return tfidf_vectorizer, tfidf_train_corpus 



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


def load_images(idata, IMG_SIZE = (224, 224)):

    paths = idata["ipath"].values
    images = [np.array(load_img(path, target_size=IMG_SIZE)) for path in paths]

    return np.array(images)

def transform_cnn(images, pretrained_model, include_top = True, layer = None, batch_size = None): 

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

    dst_file_tfidf = path.join(dst_text_features, 'tfidf_data.npy')

    
    # Generate Image-Features (VGG19) for 'Training'-data:
    
    dst_file_vgg_fc1 = path.join(dst_image_features, 'vgg_fc1_data.npy')

    print("Start: Load Images.")
    images = load_images(idata)
    print("End: Load Images.")

    vgg_fc1_features = transform_cnn(images.copy(), pretrained_model = VGG19, include_top = True, layer = "fc1") 
    save_results([vgg_fc1_features], dst_file_vgg_fc1) 
    print("VGG19 - 1")


    # Generate Text-Features (TFIDF) for 'Training' and 'Test'-data:
       
    # Set Text-Feature Parameters:
    vector_size, epochs = 4096, 100

    

    _, train_tfidf, test_tfidf = fit_tfidf(adata["text"], test_corpus = data_text["text"], max_features = vector_size)
    save_results([train_tfidf, test_tfidf], dst_file_tfidf)

   
    

