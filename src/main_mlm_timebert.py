import argparse
import logging
import random
import time
import numpy as np
import pandas as pd
import torch
from torch import optim
from torch.utils.data import DataLoader, Dataset
from transformers import BertForMaskedLM, AutoTokenizer, DataCollatorForLanguageModeling
from torch.utils.data import Dataset
from tqdm.auto import tqdm
from nltk.corpus import stopwords
import re
import string
import matplotlib.pyplot as plt


stops = set(stopwords.words('english'))

def get_mask_loc(word, text, encodings):
    start = text.find(word)
    end = start + len(word)
    label_start, label_end = 0, 0
    for i in range(len(encodings['input_ids'][0])-1):
        if encodings['offset_mapping'][0][i][0] <= start: label_start = i
        if label_end == 0 and encodings['offset_mapping'][0][i][1] >= end: label_end = i
    return label_start, label_end

class MLMDataset(Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        return item

    def __len__(self):
        return len(self.encodings["attention_mask"])

def get_best(file, metric):

    try:
        results = list()
        with open(file, 'r') as f:
            for l in f:
                if l.strip() == '':
                    continue
                t = tuple([float(v) for v in l.strip().split()])[1]
                results.append(tuple([t]))
        if metric == 'perplexity':
            return min(results)
        elif metric == 'f1':
            return max(results)

    except FileNotFoundError:
        return None

def main():
    logging.disable(logging.WARNING)
    # random.seed(123)
    # np.random.seed(123)
    # torch.manual_seed(123)

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default=None, type=str, required=True, help='Data directory')
    parser.add_argument('--results_dir', default=None, type=str, required=True, help='Results directory.')
    parser.add_argument('--seed', default=None, type=int, required=True, help='Random seed')
    parser.add_argument('--trained_dir', default=None, type=str, required=True, help='Trained model directory.')
    parser.add_argument('--batch_size', default=None, type=int, required=True, help='Batch size.')
    parser.add_argument('--lr', default=None, type=float, required=True, help='Learning rate.')
    parser.add_argument('--n_epochs', default=None, type=int, required=True, help='Number of epochs.')
    parser.add_argument('--device', default=None, type=int, required=True, help='Selected CUDA device.')
    parser.add_argument('--dataset', default=None, type=str, required=True, help='Name of dataset.')
    parser.add_argument('--year1', default=None, type=int, required=True, help='year1')
    parser.add_argument('--year2', default=None, type=int, required=True, help='year2')
    parser.add_argument('--k', default=None, type=int, required=True, help='top k')
    parser.add_argument('--method', default=None, type=int, required=True, help='method type')
    parser.add_argument('--submethod', default=None, type=int, help='submethod type')
    parser.add_argument('--template', default=None, type=int, required=True, help='template type')
    parser.add_argument('--splitter', default=None, type=str, required=True, help='nltk / sentence_splitter')
    parser.add_argument('--n_tuple', default=None, type=int, required=True, help='number of tuples')
    parser.add_argument('--checkmode', default=None, type=int, required=True, help='check mode open: 1 (Y2->Y2) or 2 (Y1->Y2) / close: 0')
    parser.add_argument('--remove_uv', default=None, type=int, required=True, help='remove u==v : 1 / 0')
    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    checkpoint = "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
      
    """ train_df = pd.read_csv('{}/yelp_train_{}.csv'.format(args.data_dir, args.year2)).loc[:,['text']]
    train_text = []
    for i in train_df['text']: train_text.append(i)
    train_encodings = tokenizer(train_text, truncation=True)
    train_dataset = MLMDataset(train_encodings) """

   
    if args.method == 3:
        word_w = pd.read_csv('{}/{}_{}_{}_w_top{}_m{}{}_{}.csv'.format(args.data_dir, args.dataset, args.year1, args.year2, args.k, args.method, args.splitter, args.submethod),header = None).iloc[0,0]
        word_u = pd.read_csv('{}/{}_{}_{}_uv_top{}_m{}{}_{}_rank_0.csv'.format(args.data_dir, args.dataset, args.year1, args.year2, args.k, args.method, args.splitter, args.submethod),header = None).iloc[:args.n_tuple*2,0].values.tolist()
        word_v = pd.read_csv('{}/{}_{}_{}_uv_top{}_m{}{}_{}_rank_0.csv'.format(args.data_dir, args.dataset, args.year1, args.year2, args.k, args.method, args.splitter, args.submethod),header = None).iloc[:args.n_tuple*2,1].values.tolist()
    else:
        word_w = pd.read_csv('{}/{}_{}_{}_w_top{}_m{}{}.csv'.format(args.data_dir, args.dataset, args.year1, args.year2, args.k, args.method, args.splitter),header = None).iloc[0,0]
        word_u = pd.read_csv('{}/{}_{}_{}_u_top{}_m{}{}.csv'.format(args.data_dir, args.dataset, args.year1, args.year2, args.k, args.method, args.splitter),header = None).iloc[:args.k,0].values.tolist()
        word_v = pd.read_csv('{}/{}_{}_{}_v_top{}_m{}{}.csv'.format(args.data_dir, args.dataset, args.year1, args.year2, args.k, args.method, args.splitter),header = None).iloc[:args.k,0].values.tolist()
    prompt_text = []
    if args.method == 3:
        cnt = 0
        i = -1
        while(cnt<args.n_tuple):
            i += 1
            if args.remove_uv == 1 and word_u[i] == word_v[i]: continue
            if args.template == 1 or (args.template == 13 and i&1 == 0) or (args.template == 12 and i&1 == 0): 
                prompt_text.append("{w} is associated with {u} in {year1}, whereas it is associated with {v} in {year2}.".format(w = word_w, u = word_u[i], v = word_v[i], year1 = args.year1, year2 = args.year2))
            elif args.template == 3 or (args.template == 13 and i&1 == 1):
                prompt_text.append("The meaning of {w} changed from {year1} to {year2} respectively from {u} to {v}.".format(w = word_w, u = word_u[i], v = word_v[i], year1 = args.year1, year2 = args.year2))
            elif args.template == 2 or (args.template == 12 and i&1 == 1):
                prompt_text.append("Unlike in {year1}, where {u} was associated with {w}, in {year2} {v} is associated with {w}.".format(w = word_w, u = word_u[i], v = word_v[i], year1 = args.year1, year2 = args.year2))
            elif args.template == 4:
                # prompt_text.append("{u} in {year1} {v} in {year2}.".format(w = word_w, u = word_u[i], v = word_v[i], year1 = args.year1, year2 = args.year2))
                prompt_text.append("{u} in {year1} and {v} in {year2}.".format(w = word_w, u = word_u[i], v = word_v[i], year1 = args.year1, year2 = args.year2))
            cnt += 1
    else:
        cnt = 0
        for i in range(int(np.sqrt(args.n_tuple))):
            for j in range(int(np.sqrt(args.n_tuple))):
                if args.remove_uv == 1 and word_u[i] == word_v[i]: continue
                if args.template == 1:
                    prompt_text.append("{w} is associated with {u} in {year1}, whereas it is associated with {v} in {year2}.".format(w = word_w, u = word_u[i], v = word_v[j], year1 = args.year1, year2 = args.year2))
                elif args.template == 3:
                    prompt_text.append("The meaning of {w} changed from {year1} to {year2} respectively from {u} to {v}.".format(w = word_w, u = word_u[i], v = word_v[j], year1 = args.year1, year2 = args.year2))
                elif args.template == 2:
                    prompt_text.append("Unlike in {year1}, where {u} was associated with {w}, in {year2} {v} is associated with {w}.".format(w = word_w, u = word_u[i], v = word_v[j], year1 = args.year1, year2 = args.year2))
                elif args.template == 4:
                    # prompt_text.append("{u} in {year1} {v} in {year2}.".format(w = word_w, u = word_u[i], v = word_v[j], year1 = args.year1, year2 = args.year2))
                    prompt_text.append("{u} in {year1} and {v} in {year2}.".format(w = word_w, u = word_u[i], v = word_v[j], year1 = args.year1, year2 = args.year2))
                #prompt_text.append("Unlike in {year1}, where {u} was associated with {w}, in {year2} {v} is associated with {w}.".format(w = word_w, u = word_u.iloc[i,0], v = word_v.iloc[j,0], year1 = args.year1, year2 = args.year2))
                #prompt_text.append("The meaning of {w} changed from {year1} to {year2} respectively from {u} to {v}.".format(w = word_w, u = word_u.iloc[i,0], v = word_v.iloc[j,0], year1 = args.year1, year2 = args.year2))
                #prompt_text.append("In {year1}, {w} was associated with {u}, whereas in {year2} it was associated with {v}.".format(w = word_w, u = word_u.iloc[i,0], v = word_v.iloc[j,0], year1 = args.year1, year2 = args.year2))
                cnt += 1
        for i in range(int(np.sqrt(args.n_tuple)), len(word_v)):
            for j in range(int(np.sqrt(args.n_tuple)), len(word_u)):
                if cnt >= args.n_tuple: break
                if args.remove_uv == 1 and word_u[i] == word_v[i]: continue
                if args.template == 1:
                    prompt_text.append("{w} is associated with {u} in {year1}, whereas it is associated with {v} in {year2}.".format(w = word_w, u = word_u[i], v = word_v[j], year1 = args.year1, year2 = args.year2))
                elif args.template == 3:
                    prompt_text.append("The meaning of {w} changed from {year1} to {year2} respectively from {u} to {v}.".format(w = word_w, u = word_u[i], v = word_v[j], year1 = args.year1, year2 = args.year2))
                elif args.template == 2:
                    prompt_text.append("Unlike in {year1}, where {u} was associated with {w}, in {year2} {v} is associated with {w}.".format(w = word_w, u = word_u[i], v = word_v[j], year1 = args.year1, year2 = args.year2))
                elif args.template == 4:
                    #reddit & YELP
                    # prompt_text.append("{u} in {year1} {v} in {year2}.".format(w = word_w, u = word_u[i], v = word_v[j], year1 = args.year1, year2 = args.year2))
                    #Arxiv
                    prompt_text.append("{u} in {year1} and {v} in {year2}.".format(w = word_w, u = word_u[i], v = word_v[j], year1 = args.year1, year2 = args.year2))
                cnt+=1
            if cnt >= args.n_tuple: break
    print("Tuple_n: {}".format(len(prompt_text)))
    print(prompt_text[:10])
    prompt_encodings = tokenizer(prompt_text, truncation=True)
    prompt_dataset = MLMDataset(prompt_encodings)

    if args.dataset == 'reddit': dev_df = pd.read_csv('{}/{}_dev_{}.csv'.format(args.data_dir, args.dataset, args.year2),lineterminator='\n').loc[:,['text']]
    else: dev_df = pd.read_csv('{}/{}_dev_{}.csv'.format(args.data_dir, args.dataset, args.year2)).loc[:,['text']]
    dev_text = []
    for i in dev_df['text']: dev_text.append(i)
    dev_encodings = tokenizer(dev_text, truncation=True)
    dev_dataset = MLMDataset(dev_encodings)

    if args.dataset == 'reddit': test_df = pd.read_csv('{}/{}_test_{}.csv'.format(args.data_dir, args.dataset, args.year2),lineterminator='\n').loc[:,['text']]
    else: test_df = pd.read_csv('{}/{}_test_{}.csv'.format(args.data_dir, args.dataset, args.year2)).loc[:,['text']]
    test_text = []
    for i in test_df['text']: test_text.append(i)
    test_encodings = tokenizer(test_text, truncation=True)
    test_dataset = MLMDataset(test_encodings)

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)
    prompt_loader = DataLoader(prompt_dataset, batch_size=args.batch_size, collate_fn=data_collator, shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=args.batch_size, collate_fn=data_collator, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, collate_fn=data_collator, shuffle=True)
    print("Data loader")

    if args.method == 3: filename = '__mlm_{}_timebert_{}_{}_batchsize_{}_top{}_method{}{}_{submethod}_T{template}_checkmode{checkmode}_{n_tuple}tuples_remove{remove}_seed{seed}_TEST0'.format(args.dataset,args.year1,args.year2,args.batch_size,args.k,args.method,args.splitter,submethod = args.submethod,checkmode = args.checkmode, n_tuple=args.n_tuple,remove=args.remove_uv,template=args.template,seed=args.seed)
    else: filename = 'mlm_{}_timebert_{}_{}_batchsize_{}_top{}_method{}{}_T{template}_checkmode{checkmode}_{n_tuple}tuples_remove{remove}'.format(args.dataset,args.year1,args.year2,args.batch_size,args.k,args.method,args.splitter,checkmode = args.checkmode, n_tuple=args.n_tuple,remove=args.remove_uv,template=args.template)

    best_result = get_best('{}/{}.txt'.format(args.results_dir, filename), metric='perplexity')
    if best_result:
        best_perplexity = best_result[0]
    else:
        best_perplexity = None
    print('Best perplexity so far: {}'.format(best_perplexity))
    device = torch.device('cuda:{}'.format(args.device) if torch.cuda.is_available() else 'cpu')
    print('Device: {}'.format(torch.cuda.get_device_name(device)))

    model = BertForMaskedLM.from_pretrained('bert-base-uncased').to(device)

    if args.checkmode == 1:
        state_dict = torch.load('{}/mlm_{}_trainedbert_{year2}_{year2}.torch'.format(args.trained_dir,args.dataset, year2 = args.year2), map_location="cuda:0")
    if args.checkmode == 2:
        state_dict = torch.load('{}/mlm_{}_trainedbert_{year1}_{year2}.torch'.format(args.trained_dir,args.dataset, year1 = args.year1, year2 = args.year2), map_location="cuda:0")
    if args.checkmode > 0: model.load_state_dict(state_dict)

    #state_dict = torch.load('{}/{}.torch'.format(args.trained_dir, filename))
    #model.load_state_dict(state_dict)
    
    #warm up + decay
    # optimizer = optim.Adam(model.parameters(), lr=0, eps=1e-06, weight_decay=0.01)

    optimizer = optim.Adam(model.parameters(), lr=args.lr, eps=1e-06, weight_decay=0.01)
    perplexity_train_ls, perplexity_dev_ls, perplexity_test_ls = [], [], []
    early_stop_count = 0
    print('Prompting model...')
    epoch_num = args.n_epochs
    for epoch in range(1, args.n_epochs + 1):

        #warm up + decay
        # if epoch <= int(epoch_num * 0.65):
        #     for params in optimizer.param_groups:
        #         params['lr'] += args.lr / (epoch_num * 0.65)
            # pass
        # elif epoch > int(epoch_num * 0.4):
        #     for params in optimizer.param_groups:
        #         params['lr'] -= args.lr / (epoch_num * 0.6)

        print(optimizer.param_groups[0]["lr"])
        print('Prompting_steps: {}'.format(len(prompt_loader)))
        progress_bar = tqdm(range(len(prompt_loader)))
        model.train()
        losses = list()
        for batch in prompt_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            losses.append(outputs.loss.item())
            loss.backward()
        
            optimizer.step()
            optimizer.zero_grad()
            progress_bar.update(1)
        perplexity_train = np.exp(np.mean(losses))
        perplexity_train_ls.append(perplexity_train)
        print('Evaluate model...')
        model.eval()

        losses = list()
        print('Eval_steps: {}'.format(len(dev_loader)))
        progress_bar = tqdm(range(len(dev_loader)))
        with torch.no_grad():
            for batch in dev_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**batch)

                losses.append(outputs.loss.item())
                progress_bar.update(1)

        perplexity_dev = np.exp(np.mean(losses))
        perplexity_dev_ls.append(perplexity_dev)

        losses = list()
        print('test_steps: {}'.format(len(test_loader)))
        progress_bar = tqdm(range(len(test_loader)))
        with torch.no_grad():
            for batch in test_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**batch)

                losses.append(outputs.loss.item())
                progress_bar.update(1)

        perplexity_test = np.exp(np.mean(losses))
        perplexity_test_ls.append(perplexity_test)
        print(epoch,perplexity_train,perplexity_dev, perplexity_test)

        with open('{}/{}.txt'.format(args.results_dir, filename), 'a+') as f:
            f.write('{}\t{}\t{}\t{:.0e}\n'.format(perplexity_train, perplexity_dev, perplexity_test, args.lr))
    
        plt.cla()
        x_axix = [i for i in range(1,epoch+1)]
        plt.title('{}_{} Result Analysis'.format(args.year1,args.year2))
        plt.plot(x_axix, perplexity_train_ls, color='green', label='train')
        plt.plot(x_axix, perplexity_dev_ls, color='red', label='dev')
        plt.plot(x_axix, perplexity_test_ls,  color='blue', label='test')
        plt.legend()
        plt.xlabel('Epochs')
        plt.ylabel('Perplexity') 
        plt.savefig('{}/{}.png'.format(args.results_dir, filename))

        if best_perplexity is None:
            best_perplexity = perplexity_dev
            early_stop_count = 0
            torch.save(model.state_dict(), '{}/{}.torch'.format(args.trained_dir, filename))
        else: 
            if best_perplexity > perplexity_dev:
                best_perplexity = perplexity_dev
                early_stop_count = 0
                torch.save(model.state_dict(), '{}/{}.torch'.format(args.trained_dir, filename))
            else:
                early_stop_count += 1
                #if early_stop_count >= 5: break
        print('Best_perplexity:{}\tStop_count:{}'.format(best_perplexity,early_stop_count))

if __name__ == '__main__':

    start_time = time.time()

    main()

    print('---------- {:.1f} minutes ----------'.format((time.time() - start_time) / 60))
    print()