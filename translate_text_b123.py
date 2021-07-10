import argparse
import pandas as pd
from tqdm import tqdm 
from google_trans_new import google_translator

'''
Example usage: python3 eng_text_b123.py 
-b1 "/home/user/SS2021/DS/DsPJ2021-Data/data/MediaEvalNewsImagesBatch01.tsv" 
-b2 "/home/user/SS2021/DS/DsPJ2021-Data/data/MediaEvalNewsImagesBatch02.tsv" 
-b3 "/home/user/SS2021/DS/DsPJ2021-Data/data/MediaEvalNewsImagesBatch03.tsv"
-o  "/home/user/SS2021/DS/DsPJ2021-Data/data/batch_01_02_03_google_texts.tsv"
'''

def parse_arguments():
  global b1_dir
  global b2_dir
  global b3_dir
  global o_dir
  parser = ArgumentParser(description='Translate article text for 3 batches')
  parser.add_argument('-b1', '--batch01_dir', help='batch01 articles directory',
                        required=True)

  parser.add_argument('-b1', '--batch01_dir', help='batch01 articles directory',
                        required=True)

  parser.add_argument('-b1', '--batch01_dir', help='batch01 articles directory',
                        required=True)
  parser.add_argument('-o', '--output_dir', help='articles with english texts directory',
                        required=True)



  args = parser.parse_args()
  b1_dir = args.batch01_dir
  b2_dir = args.batch01_dir
  b3_dir = args.batch01_dir
  o_dir = args.output_dir

def get_batches_texts(batches_dir:list):
  
  adata = pd.DataFrame(columns=['aid', 'text'])
  for b_dir in batches_dir:
    data = pd.read_csv(b_dir, delimiter="\t")
    
      
    adata.append(data[["aid", "text"]])
  return(adata)

def translate_data(adata: pd.DataFrame  ):
    L_texts=[]
    
    for de_text in tqdm(adata["text"]):
        translator=google_translator()
        en_text=translator.translate(de_text,lang_src="de",lang_tgt="en")
        L_texts.append(en_text)
        time.sleep(2)
    
        
        
    translated_texts=pd.DataFrame({"aid":adata["aid"][:],"google_text":L_texts})
    
    
    
    return(translated_texts)
    

if __name__ == "__main__":
  parse_arguments()
  adata=get_batches_texts([b1_dir,b2_dir,b3_dir])
  print(adata)
  translated_texts=translate_data(adata)
  translated_texts.to_csv(o_dir, sep='\t', encoding='utf-8', header=['aid','google_text'],index=None)
  