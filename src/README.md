**preprocess_yelp.py**, **unzip_reddit.py**, **dataget_arxiv.py**, and **dataget_ciao.py** are used to preprocess the data  
  
**get_wuv.py** and **get_wuv_roberta.py** are used to find word tuples (for BERT and RoBERTa versions)  
  
**generate.py** is used to generate auto templates  
  
**main_mlm_bert.py** and **main_mlm_roberta.py** are used to test and fine-tune the original BERT / RoBERTa  
  
**main_mlm_timebert.py** and **main_mlm_timeroberta.py** are used to test the proposed approach   


Default settting: --submetod 0 --k 100 --template 4 --remove_uv 0 --splitter sentence_splitter
