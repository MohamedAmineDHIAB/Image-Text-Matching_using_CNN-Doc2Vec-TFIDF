# Image-Text-Matching_using_CNN-Doc2Vec
#### &#x1F535; This REPO is for finding a solution to an Image-Text matching problem for german articles, using embeddings from CNN layers (VGG19) and gensim Doc2Vec for english translated articles titles as well as TF-IDF features for german articles texts limited to 255 chars.


## Requirements :
<ul>
  <li>Python2
    <ul>
      <li>gensim</li>
    </ul>
  </li>
  <li>Python3
    <ul>
      <li>google_trans_new</li>
      <li>numpy</li>
      <li>pandas</li>
      <li>spacy</li>
      <li>argparse</li>
      <li>scipy</li>
      <li>sklearn</li>
      <li>tensorflow</li>
      <li>tqdm</li>
    </ul>
  </li>
  
</ul>


 

## Steps:

#### 1. Generate tfidf text features and image features using DataProcessor.py, with following arguments:
- path to the directory containing data files containing article data (in this git repository: Image-Text-Matching_using_CNN-Doc2Vec/data)
- path to the directory containing images (folder should be crated and images downloaded, eg. Image-Text-Matching_using_CNN-Doc2Vec/image-cache)
- path to the directory where image features will be saved as vgg_fc1_data.npy (folder should be created, eg. Image-Text-Matching_using_CNN-Doc2Vec/image_features)

Results (tfidf_data.npy and vgg_fc1_data.npy) will be saved in the given as arguments locations.
Example usage: python3 DataProcessor.py -d "/Image-Text-Matching_using_CNN-Doc2Vec/data" -i "/Image-Text-Matching_using_CNN-Doc2Vec/image-cache" -tf "/Image-Text-Matching_using_CNN-Doc2Vec/text_features" -ti "/Image-Text-Matching_using_CNN-Doc2Vec/image_features"

Image feature vectors should be placed in image_features directory and can be downloaded from:

<a href="https://drive.google.com/file/d/1eNoMg-8rj8arNlnZkB0hCa7AyKxJfNTO/view?usp=sharing">Batch 01</a> 

<a href="https://drive.google.com/file/d/1FCAyzPUj1Ot2FhgHyTCW0_oqY6XPKeQ0/view?usp=sharing">Batch 02</a>

<a href="https://drive.google.com/file/d/1vW9WKkdRVjAejCsvnSRFUj5o-h0EX8hl/view?usp=sharing">Batch 03</a> 

Tfidf text features vectors should be placed in text_features directory and can be downloaded from:

<a href="https://drive.google.com/file/d/1l7gEe_bnR-ypbN44Gq1F8VQQ_JMVriiT/view?usp=sharing">Tfidf text features</a>

#### 2.Translating the texts of the articles (batch01+batch02+batch03) using eng_text_b123.py 
- run the file eng_text_b123.py with arguments : directory of batch01/ directory of batch02/ directory of batch03/ directory of Output 

Example usage: python3 eng_text_b123.py 
-b1 "Image-Text-Matching_using_CNN-Doc2Vec/data/MediaEvalNewsImagesBatch01.tsv" 
-b2 "Image-Text-Matching_using_CNN-Doc2Vec/MediaEvalNewsImagesBatch02.tsv" 
-b3 "Image-Text-Matching_using_CNN-Doc2Vec/MediaEvalNewsImagesBatch03.tsv"
-o  "Image-Text-Matching_using_CNN-Doc2Vec/batch_01_02_03_google_texts.tsv"

#### 3.Translating the titles of the articles (batch01+batch02+batch03) using deepl API:
repeat the following steps for each batch :

- import the titles alongside with the aid (for batch1/2/3) or articleID (for batch4) in one txt file 
- transform this .txt file to a .docx file (using https://convertio.co/ for example)
- import the .docx file into DeepL online Translator (https://www.deepl.com/translator) , you can divide the file into parts if you want to use the DeepL free license and wait for a cooldown after each translation , or you can simply use the premium license , in both case we get the same title translation results.

results are saved at:   

-Image-Text-Matching_using_CNN-Doc2Vec/data/batch_01_02_03_deepl_titles.tsv


#### 4. Generate doc2vec feature vectors using doc2vec_vectors.py, in a following way:
- download the doc2vec pretrained weights from the google drive link : https://drive.google.com/file/d/1813Css0589E6_SE-VJyW7GDaDiZNG2SR/view?usp=sharing and put the binary file in the folder code/data
- run the file doc2vec_vectors.py using python2 to get an array of dimension (300,13478) with 300 : a hyper parameter of the doc2vec model which corresponds to the embedding dimension and 13478 : number of articles in batches 01/02/03. The result is saved in numpy array in 'text_features/batch_01_02_03_titles_vectors.npy'

Example usage: python2 doc2vec_vectors.py

#### 5. Calculate results for similarity using DataPredictor.py, with following arguments:
- path to the directory where cosine similarity results will be saved (eg. Image-Text-Matching_using_CNN-Doc2Vec/data)
- path to the directory containing images (eg. Image-Text-Matching_using_CNN-Doc2Vec/image-cache)
- path to the directory containing text features (eg. Image-Text-Matching_using_CNN-Doc2Vec/text_features/batch_01_02_03_titles_vectors.npy)
- path to the directory containing image features (eg. Image-Text-Matching_using_CNN-Doc2Vec/image_features)

Results will be saved in the given as argument location.



#### 6.Calculate metrics using DataEvaluation.py with following arguments:
- path to directory with prediction  result and ground truth files 
- file name of the file containing predictions
- file name of the file containing ground truth


## The Metrics used in this Project are : 


<img src="https://latex.codecogs.com/png.latex?\LARGE&space;\centering&space;Accuracy@N&space;=&space;\frac{1}{K}&space;\sum_{i=1}^{K}&space;\psi_N(\textbf{image}_i)" title="\LARGE \centering Accuracy@N = \frac{1}{K} \sum_{i=1}^{K} \psi_N(\textbf{image}_i)" />

For the first metric ACC@N , the internal accuracy term is equal to 1, if the true image lies within the number of N predictions, otherwise the term is equal to 0. It is calculated for each article and added together and then averaged over the total number of articles K .

<img src="https://latex.codecogs.com/png.latex?\LARGE&space;MRR@N&space;=&space;\frac{1}{K}&space;\sum_{i=1}^{K}&space;\frac{1}{\rho\left(&space;\textbf{image}_i\right)}" title="\LARGE MRR@N = \frac{1}{K} \sum_{i=1}^{K} \frac{1}{\rho\left( \textbf{image}_i\right)}" />

The Mean Reciprocal Rank, which is a metric for evaluating ranked retrieved elements. Each prediction has a rank between 1 and N and the lower the rank the higher the score given to this prediction as a match by the model.  If the ground truth image is present in the retrieved list of size N then <img src="https://latex.codecogs.com/png.latex?\dpi{70}&space;\LARGE&space;\rho" title="\LARGE \rho" /> corresponds to it's rank. Otherwise, the internal term of the sum is set to 0 (the rank is inf).
