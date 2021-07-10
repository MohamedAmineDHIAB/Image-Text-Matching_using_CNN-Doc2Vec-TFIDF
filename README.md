# Image-Text-Matching_using_CNN-Doc2Vec
Image-Text matching using embeddings from CNN and gensim Doc2Vec


## Requirements :

- Python2
- Python3
- gensim : 'pip2 install gensim'
- google_trans_new : 'pip3 install google-trans-new'
- pandas
- numpy
- sklearn
- tensorflow
- string
- spacy
- scipy

## Steps:

#### 1. Generate tfidf text features and image features using DataProcessor.py, with following arguments:
- path to the directory containing data files containing article data (in this git repository: Image-Text-Matching_using_CNN-Doc2Vec/data)
- path to the directory containing images (folder should be crated and images downloaded, eg. Image-Text-Matching_using_CNN-Doc2Vec/image-cache)
- path to the directory where image features will be saved as vgg_fc1_data.npy (folder should be crated, eg. Image-Text-Matching_using_CNN-Doc2Vec/image_features)

Results will be saved in the given as arguments locations.

#### 2. Generate doc2vec feature vectors using doc2vec_vectors.py, in a following way:
- download the doc2vec pretrained weights from the google drive link : https://drive.google.com/file/d/1813Css0589E6_SE-VJyW7GDaDiZNG2SR/view?usp=sharing and put the binary file in the folder code/data
- run the file doc2vec_vectors.py using python2 to get an array of dimension (300,13478) with 300 : a hyper parameter of the doc2vec model which corresponds to the embedding dimension and 13478 : number of articles in batches 01/02/03. The result is saved in numpy array in 'data/batch_01_02_03_titles_vectors.npy'

Example usage: python2 doc2vec_vectors.py

#### 3. Calculate results for similarity using DataPredictor.py, with following arguments:
- path to the directory where cosine similarity results will be saved (eg. Image-Text-Matching_using_CNN-Doc2Vec/data)
- path to the directory containing images (eg. Image-Text-Matching_using_CNN-Doc2Vec/image-cache)
- - path to the directory containing text features (eg. Image-Text-Matching_using_CNN-Doc2Vec/data/batch_01_02_03_titles_vectors.npy)
- path to the directory containing image features (eg. Image-Text-Matching_using_CNN-Doc2Vec/image_features)

Results will be saved in the given as argument location.

#### 4.Translating the texts of the articles (batch01+batch02+batch03) using eng_text_b123.py 
- run the file eng_text_b123.py with arguments : directory of batch01/ directory of batch02/ directory of batch03/ directory of Output 

Example usage: python3 eng_text_b123.py 
-b1 "Image-Text-Matching_using_CNN-Doc2Vec/data/MediaEvalNewsImagesBatch01.tsv" 
-b2 "Image-Text-Matching_using_CNN-Doc2Vec/MediaEvalNewsImagesBatch02.tsv" 
-b3 "Image-Text-Matching_using_CNN-Doc2Vec/MediaEvalNewsImagesBatch03.tsv"
-o  "Image-Text-Matching_using_CNN-Doc2Vec/batch_01_02_03_google_texts.tsv"

#### 5.Translating the titles of the articles (batch01+batch02+batch03+batch04) using deepl API:
repeat the following steps for each batch :

- import the titles alongside with the aid (for batch1/2/3) or articleID (for batch4) in one txt file 
- transform this .txt file to a .docx file (using https://convertio.co/ for example)
- import the .docx file into DeepL online Translator (https://www.deepl.com/translator) , you can divide the file into parts if you want to use the DeepL free license and wait for a cooldown after each translation , or you can simply use the premium license , in both case we get the same title translation results.

results are saved at:   

-Image-Text-Matching_using_CNN-Doc2Vec/data/batch_01_02_03_deepl_titles.tsv

-Image-Text-Matching_using_CNN-Doc2Vec/data/batch_04_deepl_titles.tsv

#### 6.Calculate metrics using DataEvaluation.py with following arguments:
- path to directory with prediction  result and ground truth files 
- file name of the file containing predictions
- file name of the file containing ground truth
