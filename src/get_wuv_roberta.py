import argparse
import logging
import random
import time
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import nltk
from torch import optim
from torch.utils.data import DataLoader, Dataset
from torch.utils.data import Dataset
from tqdm.auto import tqdm
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re
import string
from sentence_splitter import SentenceSplitter, split_text_into_sentences
from transformers import RobertaTokenizer, RobertaModel, RobertaForMaskedLM

stops = set(stopwords.words('english'))
logging.disable(logging.WARNING)
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

model = RobertaModel.from_pretrained('roberta-base', output_hidden_states=True)

eval_res = model.eval()
device = 1

def get_roberta_embedding(sent):
    #sent = sent[:510] # truncation, since there are long sentences
    tokenized_text = tokenizer.tokenize("<s> {0} </s>".format(sent))
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
    segments_ids = [0 for i in range(len(indexed_tokens))]
    tokens_tensor = torch.tensor([indexed_tokens])
    segments_tensors = torch.tensor([segments_ids])
    tokens_tensor = tokens_tensor.to(device)
    segments_tensors = segments_tensors.to(device)
    model.to(device)
    with torch.no_grad():
        outputs = model(tokens_tensor, token_type_ids=segments_tensors)
    # res = list(zip(tokenized_text[1:-1], outputs[0].cpu().detach().numpy()[0][1:-1])) ## [1:-1] is used to get rid of CLS] and [SEP]
    layers_vecs = np.sum([outputs[2][-1].cpu().detach().numpy(), outputs[2][-2].cpu().detach().numpy(), outputs[2][-3].cpu().detach().numpy(), outputs[2][-4].cpu().detach().numpy()], axis=0) ### use the last 4 layers
    res = list(zip(tokenized_text[1:-1], layers_vecs[0][1:-1]))
    
    ## merge subtokens
    sent_tokens_vecs = []
    index = 0
    while index < len(res):
        token = ""
        token += res[index][0]
        token_vecs = []
        token_vecs.append(res[index][1])
        while index + 1 < len(res):
            if not res[index + 1][0].startswith('Ġ'):
                token_vecs.append(res[index+1][1])
                token += res[index+1][0]
                index += 1
            else: break
        merged_vec = np.array(token_vecs, dtype='float32').mean(axis=0) 
        merged_vec = torch.from_numpy(merged_vec).to(device)
        token = token.strip('Ġ')
        sent_tokens_vecs.append((token, merged_vec))
        index += 1
    
    return sent_tokens_vecs

def jaccard_score(a, b):
    return len(a.intersection(b))/float(len(a.union(b)))

def get_freq_w(dicty1, dicty2):
    score_w = dict()
    for key in dicty1:
        if key in stops or not key in dicty2: continue 
        if len(key) == 1: continue 
        score_w[key] = min(dicty1[key],dicty2[key])
    score_w = sorted(score_w.items(),key=lambda x:x[1],reverse=True)
    return score_w

def get_pmi(word_w, twod_dict, sent_cnt):
    pmi_wx = dict()
    count_senw = len(twod_dict[word_w])
    for key in twod_dict:
        if key == word_w: continue
        count_senx = len(twod_dict[key])
        count_sen_wx = len(twod_dict[key].intersection(twod_dict[word_w]))
        if count_sen_wx == 0: continue
        pmi_wx[key] = np.log( (count_sen_wx/float(sent_cnt)) / ( (count_senw/float(sent_cnt))*(count_senx/float(sent_cnt)) ) )
    return sorted(pmi_wx.items(),key=lambda x:x[1],reverse=True)

def get_word_w_u_v_method1(dicty1, dicty2, sent_cnty1, sent_cnty2, twod_dicty1, twod_dicty2):
    print("start find words")
    score_w = get_freq_w(dicty1, dicty2)
    word_w = score_w[0][0]

    for key in list(twod_dicty1):
        if len(twod_dicty1[key]) < 100: twod_dicty1.pop(key)

    for key in list(twod_dicty2):
        if len(twod_dicty2[key]) < 100: twod_dicty2.pop(key)

    pmi_wu = get_pmi(word_w, twod_dicty1, sent_cnty1)
    pmi_wv = get_pmi(word_w, twod_dicty2, sent_cnty2) 
    return score_w, pmi_wu, pmi_wv

def get_word_w_u_v_method2(dicty1, dicty2, sent_cnty1, sent_cnty2, twod_dicty1, twod_dicty2, topk):
    print("start find words")
    score_w = get_freq_w(dicty1, dicty2)

    for key in list(twod_dicty1):
        if len(twod_dicty1[key]) < 100: twod_dicty1.pop(key)

    for key in list(twod_dicty2):
        if len(twod_dicty2[key]) < 100: twod_dicty2.pop(key)
    jaccard_w = []
    print('steps: {}'.format(topk))
    progress_bar = tqdm(range(topk))
    for i in range(topk):
        word_w = score_w[i][0]
        set_u = {x[0] for x in get_pmi(word_w, twod_dicty1, sent_cnty1)[:topk]}
        set_v = {x[0] for x in get_pmi(word_w, twod_dicty2, sent_cnty2)[:topk]}
        jaccard_w.append([word_w, score_w[i][1], jaccard_score(set_u,set_v)])
        progress_bar.update(1)
    jaccard_w = sorted(jaccard_w,key = lambda x: x[2])
    word_w = jaccard_w[0][0]
    pmi_wu = get_pmi(word_w, twod_dicty1, sent_cnty1)
    pmi_wv = get_pmi(word_w, twod_dicty2, sent_cnty2)
    return jaccard_w, pmi_wu, pmi_wv

#implementation with poping because of GPU memory
""" def get_word_w_u_v_method3(dicty1, dicty2, twod_dicty1, twod_dicty2, embed_y1, embed_y2, topk):
    print("start find words")
    for key in list(embed_y1):
        if len(twod_dicty1[key]) < 2000: embed_y1.pop(key)
    for key in list(embed_y2):
        if len(twod_dicty2[key]) < 2000: embed_y2.pop(key)
    score_w = get_freq_w(dicty1, dicty2)
    sim_score_w = []
    year1_index = dict()
    year2_index = dict()
    cnt = 0
    for key in embed_y1.keys():
        year1_index[key] = cnt
        cnt+=1
    cnt = 0
    for key in embed_y2.keys():
        year2_index[key] = cnt
        cnt+=1
    year1_key = list(embed_y1)
    year2_key = list(embed_y2)
    embed_y1_tensor = torch.tensor([x.tolist() for x in embed_y1.values()]).to(device)
    embed_y2_tensor = torch.tensor([x.tolist() for x in embed_y2.values()]).to(device)
    sim_mat_y1 = torch.cosine_similarity(embed_y1_tensor.unsqueeze(1), embed_y1_tensor.unsqueeze(0), dim=-1)
    sim_mat_y2 = torch.cosine_similarity(embed_y2_tensor.unsqueeze(1), embed_y2_tensor.unsqueeze(0), dim=-1)
    sim_mat_y1_y2 = torch.cosine_similarity(embed_y1_tensor.unsqueeze(1), embed_y2_tensor.unsqueeze(0), dim=-1)
    print('steps: {}'.format(topk))
    progress_bar = tqdm(range(topk))
    for k in range(topk):
        sim_score = 0
        word_w = score_w[k][0]
        #if word_w not in year1_index or word_w not in year2_index: continue
        value_u,index_u = torch.topk(sim_mat_y1[year1_index[word_w]],topk)
        value_v,index_v = torch.topk(sim_mat_y2[year2_index[word_w]],topk)
        for i in range(len(value_u)):
            for j in range(len(value_v)):
                #word_u_index_y2 = year2_index[year1_key[index_u[i]]]
                #word_v_index_y1 = year1_index[year2_key[index_v[j]]]
                smi_w1_u2 = 0.0 if year1_key[index_u[i]] not in year2_index else sim_mat_y1_y2[year1_index[word_w], year2_index[year1_key[index_u[i]]]]
                smi_w2_v1 = 0.0 if year2_key[index_v[j]] not in year1_index else sim_mat_y1_y2[year1_index[year2_key[index_v[j]]], year2_index[word_w]]
                sim_score += value_u[i] + value_v[j] - smi_w1_u2 - smi_w2_v1
        sim_score_w.append([word_w, score_w[k][1], sim_score.item()])
        progress_bar.update(1)
    sim_score_w = sorted(sim_score_w,key = lambda x: x[2], reverse=True)
    word_w = score_w[0][0]
    value_u,index_u = torch.topk(sim_mat_y1[year1_index[word_w]],topk)
    value_v,index_v = torch.topk(sim_mat_y2[year2_index[word_w]],topk)
    word_uv = []
    for i in range(len(value_u)):
            for j in range(len(value_v)):
                smi_w1_u2 = 0.0 if year1_key[index_u[i]] not in year2_index else sim_mat_y1_y2[year1_index[word_w], year2_index[year1_key[index_u[i]]]]
                smi_w2_v1 = 0.0 if year2_key[index_v[j]] not in year1_index else sim_mat_y1_y2[year1_index[year2_key[index_v[j]]], year2_index[word_w]]
                uv_score = value_u[i] + value_v[j] - smi_w1_u2 - smi_w2_v1
                word_uv.append([year1_key[index_u[i]], year2_key[index_v[j]], uv_score.item()])
    word_uv = sorted(word_uv,key = lambda x: x[2], reverse=True)
    return sim_score_w, word_uv """

#implementation without poping words because of GPU Memory (not using matrix)
def get_word_w_u_v_method3(dicty1, dicty2, twod_dicty1, twod_dicty2, embed_y1, embed_y2, topk, submethod, rank):
    print("start find words")
    """ for key in list(embed_y1):
        if len(twod_dicty1[key]) < 100: embed_y1.pop(key)
    for key in list(embed_y2):
        if len(twod_dicty2[key]) < 100: embed_y2.pop(key) """
    score_w = get_freq_w(dicty1, dicty2)
    sim_score_w = []
    year1_index = dict()
    year2_index = dict()
    cnt = 0
    for key in embed_y1.keys():
        year1_index[key] = cnt
        cnt+=1
    cnt = 0
    for key in embed_y2.keys():
        year2_index[key] = cnt
        cnt+=1
    year1_key = list(embed_y1)
    year2_key = list(embed_y2)
    progress_bar = tqdm(range(topk))
    for k in range(topk):
        sim_score = 0
        word_w = score_w[k][0]
        #if word_w not in year1_index or word_w not in year2_index: continue
        word_u_ls = []
        for key, value in embed_y1.items():
            word_u_ls.append([key,torch.cosine_similarity(embed_y1[word_w],value,dim=-1)])
        word_u_ls = sorted(word_u_ls,key = lambda x: x[1], reverse=True)
        word_v_ls = []
        for key, value in embed_y2.items():
            word_v_ls.append([key,torch.cosine_similarity(embed_y2[word_w],value,dim=-1)])
        word_v_ls = sorted(word_v_ls,key = lambda x: x[1], reverse=True)
        for i in range(topk):
            for j in range(topk):
                #sim_w2_u2 = 0.0 if word_u_ls[i][0] not in embed_y2 else torch.cosine_similarity(embed_y2[word_w],embed_y2[word_u_ls[i][0]],dim=-1)
                #sim_w1_v1 = 0.0 if word_v_ls[j][0] not in embed_y1 else torch.cosine_similarity(embed_y1[word_w],embed_y1[word_v_ls[j][0]],dim=-1)
                #sim_w2_u1 = 0.0 if word_u_ls[i][0] not in embed_y1 else torch.cosine_similarity(embed_y2[word_w],embed_y1[word_u_ls[i][0]],dim=-1)
                #sim_w1_v2 = 0.0 if word_v_ls[j][0] not in embed_y2 else torch.cosine_similarity(embed_y1[word_w],embed_y2[word_v_ls[j][0]],dim=-1)
                #sim_w1_u2 = 0.0 if word_u_ls[i][0] not in embed_y2 else torch.cosine_similarity(embed_y1[word_w],embed_y2[word_u_ls[i][0]],dim=-1)
                #sim_w2_v1 = 0.0 if word_v_ls[j][0] not in embed_y1 else torch.cosine_similarity(embed_y2[word_w],embed_y1[word_v_ls[j][0]],dim=-1)
                #sim_score += word_u_ls[i][1] + word_v_ls[j][1] - sim_w1_u2 - sim_w2_v1
                if submethod == 0:
                    sim_w_u = 0.0 if word_u_ls[i][0] not in embed_y2 else torch.cosine_similarity(embed_y1[word_w],embed_y2[word_u_ls[i][0]],dim=-1)
                    sim_w_v = 0.0 if word_v_ls[j][0] not in embed_y1 else torch.cosine_similarity(embed_y2[word_w],embed_y1[word_v_ls[j][0]],dim=-1)
                elif submethod == 1:
                    sim_w_u = 0.0 if word_u_ls[i][0] not in embed_y1 else torch.cosine_similarity(embed_y2[word_w],embed_y1[word_u_ls[i][0]],dim=-1)
                    sim_w_v = 0.0 if word_v_ls[j][0] not in embed_y2 else torch.cosine_similarity(embed_y1[word_w],embed_y2[word_v_ls[j][0]],dim=-1)
                elif submethod == 2:
                    sim_w_u = 0.0 if word_u_ls[i][0] not in embed_y2 else torch.cosine_similarity(embed_y2[word_w],embed_y2[word_u_ls[i][0]],dim=-1)
                    sim_w_v = 0.0 if word_v_ls[j][0] not in embed_y1 else torch.cosine_similarity(embed_y1[word_w],embed_y1[word_v_ls[j][0]],dim=-1)
                sim_score += word_u_ls[i][1] + word_v_ls[j][1] - sim_w_u - sim_w_v
        sim_score_w.append([word_w, score_w[k][1], sim_score.item()])
        progress_bar.update(1)
    sim_score_w = sorted(sim_score_w,key = lambda x: x[2], reverse=True)
    #word_w = score_w[0][0]
    word_uv_ls = []
    progress_bar = tqdm(range(rank))
    for r in range(rank):
        word_w = sim_score_w[r][0]
        word_u_ls = []
        for key, value in embed_y1.items():
            word_u_ls.append([key,torch.cosine_similarity(embed_y1[word_w],value,dim=-1)])
        word_u_ls = sorted(word_u_ls,key = lambda x: x[1], reverse=True)
        word_v_ls = []
        for key, value in embed_y2.items():
            word_v_ls.append([key,torch.cosine_similarity(embed_y2[word_w],value,dim=-1)])
        word_v_ls = sorted(word_v_ls,key = lambda x: x[1], reverse=True)
        word_uv = []
        for i in range(topk):
            for j in range(topk):
                #sim_w2_u2 = 0.0 if word_u_ls[i][0] not in embed_y2 else torch.cosine_similarity(embed_y2[word_w],embed_y2[word_u_ls[i][0]],dim=-1)
                #sim_w1_v1 = 0.0 if word_v_ls[j][0] not in embed_y1 else torch.cosine_similarity(embed_y1[word_w],embed_y1[word_v_ls[j][0]],dim=-1)
                #sim_w2_u1 = 0.0 if word_u_ls[i][0] not in embed_y1 else torch.cosine_similarity(embed_y2[word_w],embed_y1[word_u_ls[i][0]],dim=-1)
                #sim_w1_v2 = 0.0 if word_v_ls[j][0] not in embed_y2 else torch.cosine_similarity(embed_y1[word_w],embed_y2[word_v_ls[j][0]],dim=-1)
                #sim_w1_u2 = 0.0 if word_u_ls[i][0] not in embed_y2 else torch.cosine_similarity(embed_y1[word_w],embed_y2[word_u_ls[i][0]],dim=-1)
                #sim_w2_v1 = 0.0 if word_v_ls[j][0] not in embed_y1 else torch.cosine_similarity(embed_y2[word_w],embed_y1[word_v_ls[j][0]],dim=-1)
                #uv_score = word_u_ls[i][1] + word_v_ls[j][1] - sim_w1_u2 - sim_w2_v1
                if submethod == 0:
                    sim_w_u = 0.0 if word_u_ls[i][0] not in embed_y2 else torch.cosine_similarity(embed_y1[word_w],embed_y2[word_u_ls[i][0]],dim=-1)
                    sim_w_v = 0.0 if word_v_ls[j][0] not in embed_y1 else torch.cosine_similarity(embed_y2[word_w],embed_y1[word_v_ls[j][0]],dim=-1)
                elif submethod == 1:
                    sim_w_u = 0.0 if word_u_ls[i][0] not in embed_y1 else torch.cosine_similarity(embed_y2[word_w],embed_y1[word_u_ls[i][0]],dim=-1)
                    sim_w_v = 0.0 if word_v_ls[j][0] not in embed_y2 else torch.cosine_similarity(embed_y1[word_w],embed_y2[word_v_ls[j][0]],dim=-1)
                elif submethod == 2:
                    sim_w_u = 0.0 if word_u_ls[i][0] not in embed_y2 else torch.cosine_similarity(embed_y2[word_w],embed_y2[word_u_ls[i][0]],dim=-1)
                    sim_w_v = 0.0 if word_v_ls[j][0] not in embed_y1 else torch.cosine_similarity(embed_y1[word_w],embed_y1[word_v_ls[j][0]],dim=-1)
                uv_score = word_u_ls[i][1] + word_v_ls[j][1] - sim_w_u - sim_w_v
                word_uv.append([word_u_ls[i][0], word_v_ls[j][0], uv_score.item()])        
        word_uv = sorted(word_uv,key = lambda x: x[2], reverse=True)
        word_uv_ls.append(word_uv)
        progress_bar.update(1)
    return sim_score_w, word_uv_ls

def get_embedding(twod_dict,sent_ls,args):
    emb_dict = dict()
    num_dict = dict()
    for key in list(twod_dict):
        #if args.dataset == 'yelp':
        if len(twod_dict[key]) < 100: twod_dict.pop(key)
        #if args.dataset == 'reddit':
            #if len(twod_dict[key]) < 100: twod_dict.pop(key)
    print('word_num: {}'.format(len(twod_dict)))
    print('steps:{}'.format(len(sent_ls)))
    progress_bar = tqdm(range(len(sent_ls)))
    for sent in sent_ls:
        """ with open('data/test.txt', 'w+') as f:
            f.write(sent)  """
        embedding = get_roberta_embedding(sent)
        for t in embedding:
            if str(t[0]) not in twod_dict: continue
            if str(t[0]) not in emb_dict:
                emb_dict[str(t[0])] = t[1]
                num_dict[str(t[0])] = torch.tensor([1.]).to(device)
            else:
                emb_dict[str(t[0])] += t[1]
                num_dict[str(t[0])] += 1
        progress_bar.update(1)
    for key in emb_dict.keys():
        emb_dict[key] = emb_dict[key].div(num_dict[key])
    #print(emb_dict)
    #emb_tensor = torch.stack(tuple([x for x in emb_dict.values()]),dim=0)
    #num_tensor = torch.stack(tuple([x for x in num_dict.values()]),dim=0)
    #emb_tensor.div(num_tensor)
    return emb_dict

""" def show_histogram()
    sent_len_ls = []
    for sent in sent_ls:
        sent_len_ls.append(len(sent))
    fig = sns.distplot(sent_len_ls)
    fig_file = fig.get_figure()
    fig_file.savefig("output.png")"""

def get_sent_vocab(yeardf, splitter, method_type):
    if method_type == 4: sentence_ls = []
    tempdict = dict()
    numofsent_has_word = {} 
    twod_dict = {}
    sent_cnt = 0
    print('steps:{}'.format(len(yeardf['text'])))
    progress_bar = tqdm(range(len(yeardf['text']))) 
    if splitter == 'nltk': sen_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle') # use nltk tokenizer
    for i in yeardf['text']:
        if splitter == 'nltk': x = sen_tokenizer.tokenize(i) # use nltk tokenizer
        else : x = split_text_into_sentences(i, language='en') # use sentence_splitter
        for sentence in x:
            sentence = sentence.replace('\u200D' or '\u200C' or '\u200B',' ')
            sentence = sentence.encode("ascii", "ignore").decode()
            sentence = sentence.replace(chr(30),'')
            if len(sentence) > 0:
                spl_sent = []
                while(len(sentence)>500):
                    spl_sent.append(sentence[:500])
                    sentence = sentence[500:]
                spl_sent.append(sentence) 
                for sent in spl_sent:
                    sent = sent.strip()
                    if len(sent) <= 1: continue
                    if method_type == 4: sentence_ls.append(sent)
                    sent_cnt += 1
                    for word in re.findall("\w+",sent):
                        if word in tempdict: tempdict[word] += 1
                        else:
                            tempdict[word] = 1 
                            twod_dict[word] = set()
                        twod_dict[word].add(sent_cnt-1)
        progress_bar.update(1)
    if method_type == 4: return sentence_ls, sent_cnt, tempdict, twod_dict
    return sent_cnt, tempdict, twod_dict

def try_get_sent_vocab(args, year, yeardf, splitter, method_type):
    try: 
        with open('{}/{}_worddict_{}_{}.pkl'.format(args.data_dir,args.dataset, year, args.splitter), 'rb') as f:
            tempdict = pickle.load(f)
        with open('{}/{}_twod_dict_{}_{}.pkl'.format(args.data_dir,args.dataset, year, args.splitter), 'rb') as f:
            twod_dict = pickle.load(f)
        with open('{}/{}_sent_cnt_{}_{}.pkl'.format(args.data_dir,args.dataset, year, args.splitter), 'rb') as f:
            sent_cnt = pickle.load(f)    
        return sent_cnt, tempdict, twod_dict
    except FileNotFoundError:
        sent_cnt, tempdict, twod_dict = get_sent_vocab(yeardf, splitter, method_type)
        with open('{}/{}_worddict_{}_{}.pkl'.format(args.data_dir,args.dataset, year, args.splitter), 'wb') as f:
            pickle.dump(tempdict, f)
        with open('{}/{}_twod_dict_{}_{}.pkl'.format(args.data_dir,args.dataset, year, args.splitter), 'wb') as f:
            pickle.dump(twod_dict, f)
        with open('{}/{}_sent_cnt_{}_{}.pkl'.format(args.data_dir,args.dataset, year, args.splitter), 'wb') as f:
            pickle.dump(sent_cnt, f)
        return sent_cnt, tempdict, twod_dict
           

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default=None, type=str, required=True, help='Data directory')
    parser.add_argument('--results_dir', default=None, type=str, required=True, help='Results directory')
    parser.add_argument('--year1', default=None, type=int, required=True, help='year1')
    parser.add_argument('--year2', default=None, type=int, required=True, help='year2')
    parser.add_argument('--dataset', default=None, type=str, required=True, help='Name of dataset.')
    parser.add_argument('--k', default=None, type=int, required=True, help='top k')
    parser.add_argument('--method', default=None, type=int, required=True, help='method type')
    parser.add_argument('--submethod', default=None, type=int, help='method type')
    parser.add_argument('--splitter', default=None, type=str, required=True, help='nltk / sentence_splitter')
    parser.add_argument('--device', default=None, type=int, required=True, help='Selected CUDA device.')
    args = parser.parse_args()
    device = args.device
    if args.splitter != 'nltk' and args.splitter != 'sentence_splitter':
        print("error\n--splitter nltk / sentence_splitter")
        return
    if args.dataset == 'reddit':
        year1_df = pd.read_csv('{}/{}_train_{}.csv'.format(args.data_dir,args.dataset, args.year1),lineterminator='\n').loc[:,['text']]
        train_df = pd.read_csv('{}/{}_train_{}.csv'.format(args.data_dir,args.dataset, args.year2),lineterminator='\n').loc[:,['text']]
    else:
        year1_df = pd.read_csv('{}/{}_train_{}.csv'.format(args.data_dir,args.dataset, args.year1)).loc[:,['text']]
        train_df = pd.read_csv('{}/{}_train_{}.csv'.format(args.data_dir,args.dataset, args.year2)).loc[:,['text']]
    if args.method == 4: year1_sent_ls, year1_sent_cnt, year1_vocab, year1_twod_dict = get_sent_vocab(year1_df, args.splitter, args.method)
    else: 
        year1_sent_cnt, year1_vocab, year1_twod_dict = try_get_sent_vocab(args, args.year1, year1_df, args.splitter, args.method)
        #year1_sent_cnt, year1_vocab, year1_twod_dict = get_sent_vocab(year1_df, args.splitter, args.method)
    print("year1 finished")
    if args.method == 4: train_sent_ls, train_sent_cnt, train_vocab, train_twod_dict = get_sent_vocab(train_df, args.splitter, args.method)
    else: 
        train_sent_cnt, train_vocab, train_twod_dict = try_get_sent_vocab(args, args.year2, train_df, args.splitter, args.method)
        #train_sent_cnt, train_vocab, train_twod_dict = get_sent_vocab(train_df, args.splitter, args.method)
    print("year2 finished")

    if args.method == 1:
        word_w_ls, word_u_ls, word_v_ls = get_word_w_u_v_method1(year1_vocab, train_vocab, year1_sent_cnt, train_sent_cnt, year1_twod_dict, train_twod_dict)
    elif args.method == 2:
        word_w_ls, word_u_ls, word_v_ls = get_word_w_u_v_method2(year1_vocab, train_vocab, year1_sent_cnt, train_sent_cnt, year1_twod_dict, train_twod_dict, args.k)
    elif args.method == 3:
        with open('{}/{}_embedding_{}_m4{}.pkl'.format(args.data_dir, args.dataset, args.year1, args.splitter), 'rb') as f:
            year1_embedding = pickle.load(f)
        with open('{}/{}_embedding_{}_m4{}.pkl'.format(args.data_dir, args.dataset, args.year2, args.splitter), 'rb') as f:
            train_embedding = pickle.load(f)
            print(len(year1_embedding),len(train_embedding))
        word_w_ls, word_uv_ls = get_word_w_u_v_method3(year1_vocab, train_vocab, year1_twod_dict, train_twod_dict, year1_embedding, train_embedding, args.k, args.submethod, args.k)
        for i in range(args.k):
            with open('{}/{}_{}_{}_uv_top{}_m{}{}_{}_rank_{}.csv'.format(args.data_dir, args.dataset, args.year1, args.year2, args.k, args.method, args.splitter, args.submethod,i), 'w+') as f:
                for uv in word_uv_ls[i][:args.k*args.k]:
                    f.write('{},{},{}\n'.format(uv[0],uv[1],uv[2]))
        with open('{}/{}_{}_{}_w_top{}_m{}{}_{}.csv'.format(args.data_dir, args.dataset, args.year1, args.year2, args.k, args.method, args.splitter, args.submethod), 'w+') as f:
            for w in word_w_ls[:args.k]:
                f.write('{},{},{}\n'.format(w[0],w[1],w[2]))
        return
    elif args.method == 4:
        """ year1_embedding_dict = get_embedding(year1_twod_dict,year1_sent_ls)
        with open('{}/embedding_{}_m{}{}.pkl'.format(args.data_dir, args.year1, args.method, args.splitter), 'wb') as f:
            pickle.dump(year1_embedding_dict, f) """
        """ with open('data/test.txt', 'w+') as f:
            f.write(train_sent_ls[319345])  """
        train_embedding_dict = get_embedding(train_twod_dict,train_sent_ls,args)
        with open('{}/{}_embedding_{}_m{}{}.pkl'.format(args.data_dir, args.dataset, args.year2, args.method, args.splitter), 'wb') as f:
            pickle.dump(train_embedding_dict, f)
        return


    with open('{}/{}_{}_{}_w_top{}_m{}{}.csv'.format(args.data_dir, args.dataset, args.year1, args.year2, args.k, args.method, args.splitter), 'w+') as f:
        for w in word_w_ls[:args.k]:
            if args.method == 1:
                f.write('{},{}\n'.format(w[0],w[1]))
            elif args.method == 2:
                f.write('{},{},{}\n'.format(w[0],w[1],w[2]))
    with open('{}/{}_{}_{}_u_top{}_m{}{}.csv'.format(args.data_dir, args.dataset, args.year1, args.year2, args.k, args.method, args.splitter), 'w+') as f:
        for u in word_u_ls[:args.k]:
            f.write('{},{}\n'.format(u[0],u[1]))
    with open('{}/{}_{}_{}_v_top{}_m{}{}.csv'.format(args.data_dir, args.dataset, args.year1, args.year2, args.k, args.method, args.splitter), 'w+') as f:
        for v in word_v_ls[:args.k]:
            f.write('{},{}\n'.format(v[0],v[1]))
    """ with open('{}/{}_{}_uv.csv'.format(args.data_dir, args.year1, args.year2), 'w+') as f:
        for i in range(min(10000,min(len(word_u_ls),len(word_v_ls)))):
            f.write('{},{}\n'.format(word_u_ls[i][0],word_v_ls[i][0])) """
    print("finished")
    


if __name__ == '__main__':

    start_time = time.time()

    main()

    print('---------- {:.1f} minutes ----------'.format((time.time() - start_time) / 60))
    print()