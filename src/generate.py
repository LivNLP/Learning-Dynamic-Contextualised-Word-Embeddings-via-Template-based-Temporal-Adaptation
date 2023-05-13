import pandas as pd
import numpy as np
import argparse
import time
import torch
from tqdm.auto import tqdm
from sentence_splitter import SentenceSplitter, split_text_into_sentences
from transformers import T5Tokenizer, T5ForConditionalGeneration
import nltk
import re

tokenizer = T5Tokenizer.from_pretrained("t5-base")
model = T5ForConditionalGeneration.from_pretrained("t5-base")
device = 0

def generate(word_u,word_v,sent1,sent2,args):
    model.to(device)
    target_number=3
    beam = 100
    length_limit = None
    input_tensors = []
    max_length = 0
    for i in range(len(sent1)):
        #input_text = sent1[i] + " " + word_u[i] + " <extra_id_0> " + str(args.year1) + " " + word_v[i] + " <extra_id_1> " + str(args.year2) + " " + sent2[i]
        input_text = sent1[i] + " " + word_u[i] + " <extra_id_0> " + str(args.year1) + " " + word_v[i] + " <extra_id_1> " + str(args.year2) + " <extra_id_2> " + sent2[i]
        #new format 
        #input_text = sent1[i] + " " + word_u[i] + " <extra_id_0> " + str(args.year1) + " <extra_id_1> " + word_v[i] + " <extra_id_2> " + str(args.year2) + " <extra_id_3> " + sent2[i]
        #max
        #input_text = sent1[i] + " <extra_id_0> " + word_u[i] + " <extra_id_1> " + str(args.year1) + " <extra_id_2> " + word_v[i] + " <extra_id_3> " + str(args.year2) + " <extra_id_4> " + sent2[i]
        input_text = tokenizer.encode(input_text)
        input_ids = torch.tensor(input_text).long()
        max_length = max(max_length, input_ids.size(-1))
        input_tensors.append(input_ids)

    """ input_ids = torch.tensor(input_text).long()
    max_length = max(max_length, input_ids.size(-1))
    input_tensors.append(input_ids) """

    # Concatenate inputs as a batch
    input_ids = torch.zeros((len(input_tensors), max_length)).long()
    attention_mask = torch.zeros((len(input_tensors), max_length)).long()
    for i in range(len(input_tensors)):
        input_ids[i, :input_tensors[i].size(-1)] = input_tensors[i]
        attention_mask[i, :input_tensors[i].size(-1)] = 1

    # Print some examples
    print('####### example #######')
    print(tokenizer.decode(input_ids[0]))
    print(tokenizer.decode(input_ids[1]))
    print(tokenizer.decode(input_ids[2]))
    print('####### example #######\n')
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.cuda()
    assert len(input_tensors) > 0
    # Maximum generate content length
    max_length = 20

    start_mask = tokenizer._convert_token_to_id('<extra_id_0>')
    ori_decoder_input_ids = torch.zeros((input_ids.size(0), max_length)).long()
    ori_decoder_input_ids[..., 0] = model.config.decoder_start_token_id

    # decoder_input_ids: decoder inputs for next regressive generation
    # ll: log likelihood
    # output_id: which part of generated contents we are at
    # output: generated content so far
    # last_length (deprecated): how long we have generated for this part
    current_output = [{'decoder_input_ids': ori_decoder_input_ids, 'll': 0, 'output_id': 1, 'output': [], 'last_length': -1}]
    for i in tqdm(range(max_length - 2)):
        new_current_output = []
        for item in current_output:
            if item['output_id'] > target_number:
                # Enough contents
                new_current_output.append(item)
                continue
            decoder_input_ids = item['decoder_input_ids']

            # Forward
            batch_size = 32
            turn = input_ids.size(0) // batch_size
            if input_ids.size(0) % batch_size != 0:
                turn += 1
            aggr_output = []
            for t in range(turn):
                start = t * batch_size
                end = min((t + 1) * batch_size, input_ids.size(0))

                with torch.no_grad():
                    aggr_output.append(model(input_ids.to(device)[start:end], attention_mask=attention_mask.to(device)[start:end], decoder_input_ids=decoder_input_ids.to(device)[start:end])[0])
            aggr_output = torch.cat(aggr_output, 0)

            # Gather results across all input sentences, and sort generated tokens by log likelihood
            aggr_output = aggr_output.mean(0)
            log_denominator = torch.logsumexp(aggr_output[i], -1).item()
            ids = list(range(model.config.vocab_size))
            ids.sort(key=lambda x: aggr_output[i][x].item(), reverse=True)
            ids = ids[:beam+3]
            
            for word_id in ids:
                output_id = item['output_id']

                if word_id == start_mask - output_id or word_id == tokenizer._convert_token_to_id('</s>'):
                    # Finish one part
                    if length_limit is not None and item['last_length'] < length_limit[output_id - 1]:
                        check = False
                    else:
                        check = True
                    output_id += 1
                    last_length = 0
                else:
                    last_length = item['last_length'] + 1
                    check = True

                output_text = item['output'] + [word_id]
                ll = item['ll'] + aggr_output[i][word_id] - log_denominator
                new_decoder_input_ids = decoder_input_ids.new_zeros(decoder_input_ids.size())
                new_decoder_input_ids[:] = decoder_input_ids
                new_decoder_input_ids[..., i + 1] = word_id

                # Forbid single space token, "....", and ".........."
                if word_id in [3, 19794, 22354]:
                    check = False

                # Forbid continuous "."
                if len(output_text) > 1 and output_text[-2] == 5 and output_text[-1] == 5:
                    check = False

                if check:
                    # Add new results to beam search pool
                    new_item = {'decoder_input_ids': new_decoder_input_ids, 'll': ll, 'output_id': output_id, 'output': output_text, 'last_length': last_length}
                    new_current_output.append(new_item)

        if len(new_current_output) == 0:
            break

        new_current_output.sort(key=lambda x: x['ll'], reverse=True)
        new_current_output = new_current_output[:beam]
        current_output = new_current_output

    print("####### generated results #######")
    for item in current_output:
        generate_text = ''
        for token in item['output']:
            generate_text += tokenizer._convert_id_to_token(token)
        print('--------------')
        print('generated text:{text} score:{score}'.format(text=generate_text,score=item['ll'].item()))
        with open('{}/{}_prompts_m{}_{}_3slots'.format(args.results_dir,args.dataset,args.method,args.n_tuple), 'a+') as f:
            #f.write("{},{},{}\n".format(word_u,word_v,generate_text))
            f.write("{},{}\n".format(generate_text,item['ll'].item()))
    print("####### generated results #######\n")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default=None, type=str, required=True, help='Data directory')
    parser.add_argument('--results_dir', default=None, type=str, required=True, help='Results directory.')
    parser.add_argument('--device', default=None, type=int, required=True, help='Selected CUDA device.')
    parser.add_argument('--dataset', default=None, type=str, required=True, help='Name of dataset.')
    parser.add_argument('--year1', default=None, type=int, required=True, help='year1')
    parser.add_argument('--year2', default=None, type=int, required=True, help='year2')
    parser.add_argument('--k', default=None, type=int, required=True, help='top k')
    parser.add_argument('--method', default=None, type=int, required=True, help='method type')
    parser.add_argument('--submethod', default=None, type=int, help='submethod type')
    parser.add_argument('--n_tuple', default=None, type=int, required=True, help='number of tuples')
    parser.add_argument('--splitter', default=None, type=str, required=True, help='nltk / sentence_splitter')

    args = parser.parse_args()
    year1_df = pd.read_csv('{}/{}_train_{}.csv'.format(args.data_dir,args.dataset,args.year1),lineterminator='\n').loc[:,['text']]
    year2_df = pd.read_csv('{}/{}_train_{}.csv'.format(args.data_dir,args.dataset,args.year2),lineterminator='\n').loc[:,['text']]
    if args.method == 3:
        word_w = pd.read_csv('{}/{}_{}_{}_w_top{}_m{}{}_{}.csv'.format(args.data_dir, args.dataset, args.year1, args.year2, args.k, args.method, args.splitter, args.submethod),header = None).iloc[0,0]
        word_u = pd.read_csv('{}/{}_{}_{}_uv_top{}_m{}{}_{}_rank_0.csv'.format(args.data_dir, args.dataset, args.year1, args.year2, args.k, args.method, args.splitter, args.submethod),header = None).iloc[:args.n_tuple,0].values.tolist()
        word_v = pd.read_csv('{}/{}_{}_{}_uv_top{}_m{}{}_{}_rank_0.csv'.format(args.data_dir, args.dataset, args.year1, args.year2, args.k, args.method, args.splitter, args.submethod),header = None).iloc[:args.n_tuple,1].values.tolist()
    else:
        word_w = pd.read_csv('{}/{}_{}_{}_w_top{}_m{}{}.csv'.format(args.data_dir, args.dataset, args.year1, args.year2, args.k, args.method, args.splitter),header = None).iloc[0,0]
        ori_word_u = pd.read_csv('{}/{}_{}_{}_u_top{}_m{}{}.csv'.format(args.data_dir, args.dataset, args.year1, args.year2, args.k, args.method, args.splitter),header = None).iloc[:args.k,0].values.tolist()
        ori_word_v = pd.read_csv('{}/{}_{}_{}_v_top{}_m{}{}.csv'.format(args.data_dir, args.dataset, args.year1, args.year2, args.k, args.method, args.splitter),header = None).iloc[:args.k,0].values.tolist()
        cnt = 0
        word_u = []
        word_v = []
        for i in range(int(np.sqrt(args.n_tuple)) + 1):
            for j in range(int(np.sqrt(args.n_tuple)) + 1):
                word_u.append(ori_word_u[i])
                word_v.append(ori_word_v[j])
                cnt += 1
                if cnt >= args.n_tuple: break
            if cnt >= args.n_tuple: break
    sen_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    device = args.device
    sent1 = []
    sent2 = []
    for i in range(args.n_tuple):
        temp1 = ""
        temp2 = ""
        flag = 0
        for txt in year1_df['text']:
            if flag: break
            #sents = split_text_into_sentences(txt, language='en')
            sents = sen_tokenizer.tokenize(txt)
            for sent in sents:
                if word_u[i] in re.findall("\w+",sent) and len(sent)<=300:
                    #print(sent)
                    with open('{}sentu_{}.txt'.format(args.dataset,args.n_tuple), 'a+') as f:
                        f.write("{}\n".format(sent))
                    flag = 1
                    temp1 = sent
                    break
        if flag == 0: 
            word_u.pop(i)
            word_v.pop(i)
            continue
        flag = 0
        for txt in year2_df['text']:
            if flag: break
            #sents = split_text_into_sentences(txt, language='en')
            sents = sen_tokenizer.tokenize(txt)
            for sent in sents:   
                if word_v[i] in re.findall("\w+",sent) and len(sent)<=300:
                    #print(sent)
                    with open('{}sentv_{}.txt'.format(args.dataset,args.n_tuple), 'a+') as f:
                        f.write("{}\n".format(sent))
                    flag = 1
                    temp2 = sent
                    break
        if flag == 0:
            word_u.pop(i)
            word_v.pop(i)
            continue
        sent1.append(temp1)
        sent2.append(temp2)
    print(len(word_u),len(word_v),len(sent1),len(sent2))
    assert len(word_v) == len(sent1)
    generate(word_u,word_v,sent1,sent2,args)

if __name__ == '__main__':

    start_time = time.time()

    main()

    print('---------- {:.1f} minutes ----------'.format((time.time() - start_time) / 60))
    print()