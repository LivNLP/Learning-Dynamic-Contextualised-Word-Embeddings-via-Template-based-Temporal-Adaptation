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
import matplotlib.pyplot as plt



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
    random.seed(123)
    np.random.seed(123)
    torch.manual_seed(123)

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default=None, type=str, required=True, help='Data directory')
    parser.add_argument('--results_dir', default=None, type=str, required=True, help='Results directory.')
    parser.add_argument('--trained_dir', default=None, type=str, required=True, help='Trained model directory.')
    parser.add_argument('--batch_size', default=None, type=int, required=True, help='Batch size.')
    parser.add_argument('--lr', default=None, type=float, required=True, help='Learning rate.')
    parser.add_argument('--n_epochs', default=None, type=int, required=True, help='Number of epochs.')
    parser.add_argument('--device', default=None, type=int, required=True, help='Selected CUDA device.')
    parser.add_argument('--dataset', default=None, type=str, required=True, help='Name of dataset.')
    parser.add_argument('--year1', default=None, type=int, required=True, help='The year1')
    parser.add_argument('--year2', default=None, type=int, required=True, help='The year2')
    parser.add_argument('--checkmode', default=None, type=int, required=True, help='checkmode 0/1')
    args = parser.parse_args()
    checkpoint = "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    
    if args.checkmode == 1: 
        if args.dataset == 'reddit': train_df = pd.read_csv('{}/{}_train_{}.csv'.format(args.data_dir, args.dataset, args.year1),lineterminator='\n').loc[:,['text']]
        else: train_df = pd.read_csv('{}/{}_train_{}.csv'.format(args.data_dir, args.dataset, args.year1)).loc[:,['text']]
        train_text = []
        for i in train_df['text']: train_text.append(i)
        train_encodings = tokenizer(train_text, truncation=True)
        train_dataset = MLMDataset(train_encodings)

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
    if args.checkmode == 1: train_loader = DataLoader(train_dataset, batch_size=args.batch_size, collate_fn=data_collator, shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=args.batch_size, collate_fn=data_collator, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, collate_fn=data_collator, shuffle=True)
    print("Data loader")
    if args.checkmode == 1: filename = 'mlm_{}_trainedbert_{}_{}'.format(args.dataset,args.year1,args.year2)
    else: filename = 'mlm_{}_bert_{}_{}'.format(args.dataset,args.year1,args.year2)
    best_result = get_best('{}/{}.txt'.format(args.results_dir, filename), metric='perplexity')
    if best_result:
        best_perplexity = best_result[0]
    else:
        best_perplexity = None
    print('Best perplexity so far: {}'.format(best_perplexity))
    device = torch.device('cuda:{}'.format(args.device) if torch.cuda.is_available() else 'cpu')

    model = BertForMaskedLM.from_pretrained(checkpoint).to(device)

    #state_dict = torch.load('{}/mlm_yelp_trainedbert_{year1}_{year2}.torch'.format(args.trained_dir, year1 = args.year1, year2 = args.year2))
    #model.load_state_dict(state_dict)
    if args.checkmode == 1:
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        num_training_steps = args.n_epochs * len(train_loader)
        print('Trainning_steps: {}'.format(num_training_steps))

    print('Train model...')
    perplexity_train_ls, perplexity_dev_ls, perplexity_test_ls = [], [], []
    early_stop_count = 0 
    for epoch in range(1,1+args.n_epochs):
        if args.checkmode == 1:
            progress_bar = tqdm(range(len(train_loader)))
            model.train()
            losses = list()
            for batch in train_loader:
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

        with torch.no_grad():
            for batch in dev_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**batch)

                losses.append(outputs.loss.item())

        perplexity_dev = np.exp(np.mean(losses))
        perplexity_dev_ls.append(perplexity_dev)
        losses = list()

        with torch.no_grad():
            for batch in test_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**batch)

                losses.append(outputs.loss.item())

        perplexity_test = np.exp(np.mean(losses))
        perplexity_test_ls.append(perplexity_test)
        if args.checkmode == 1: print(epoch, perplexity_train,perplexity_dev, perplexity_test)
        else: print(epoch,perplexity_dev, perplexity_test)
        with open('{}/{}.txt'.format(args.results_dir, filename), 'a+') as f:
            if args.checkmode == 1:
                f.write('{}\t{}\t{}\t{:.0e}\n'.format(perplexity_train, perplexity_dev, perplexity_test, args.lr))
            else: f.write('{}\t{}\t{:.0e}\n'.format(perplexity_dev, perplexity_test, args.lr))
        if args.checkmode == 1:
            plt.cla()
            x_axix = [i for i in range(1, epoch + 1)]
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
                torch.save(model.state_dict(), '{}/{}.torch'.format(args.trained_dir, filename))
            else: 
                if best_perplexity > perplexity_dev:
                    best_perplexity = perplexity_dev
                    early_stop_count = 0
                    torch.save(model.state_dict(), '{}/{}.torch'.format(args.trained_dir, filename))
                    #compare with DCWE
                    #if perplexity_test < 4.720: break
                else:
                    early_stop_count += 1
                    if early_stop_count >= 5: break
            print('Best_perplexity:{}\tStop_count:{}'.format(best_perplexity,early_stop_count))
            
        

if __name__ == '__main__':

    start_time = time.time()

    main()

    print('---------- {:.1f} minutes ----------'.format((time.time() - start_time) / 60))
    print()