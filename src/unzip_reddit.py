import zstandard as zstd
import json
import pandas as pd
import time
import argparse
from sklearn.model_selection import train_test_split

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--year', default=None, type=int, required=True, help='year')
    parser.add_argument('--month', default=None, type=int, required=True, help='month')
    args = parser.parse_args()

    data = pd.DataFrame(data=None,columns=['body','author_created_utc','subreddit'])
    subdict = dict()
    with open("RC_{}-{}.zst".format(args.year,args.month if args.month >= 10 else '0'+str(args.month)), 'rb') as fh:
        dctx = zstd.ZstdDecompressor(max_window_size=2**31)
        with dctx.stream_reader(fh) as reader:
            previous_line = ""
            cnt = 0
            while True:
                chunk = reader.read(2**27)
                if not chunk:
                    break
                temp = []
                string_data = chunk.decode('utf-8')
                lines = string_data.split("\n")
                for i, line in enumerate(lines[:-1]):
                    if i == 0:
                        line = previous_line + line
                    c = json.loads(line)
                    ##start dealing with json objects
                    for k in list(c):
                        if k != 'body' and k != 'author_created_utc' and k != 'subreddit':
                            del(c[k])
                    temp.append(c)
                    cnt += 1
                    with open('cnt_{}.txt'.format(args.year), 'w+') as f:#to check progress
                        f.write('{}'.format(cnt))
                previous_line = lines[-1]
                data = pd.concat([data,pd.DataFrame(temp)])
    # filter data                  
    data.rename(columns={'body': 'text', 'author_created_utc': 'user'}, inplace=True)
    data = data[data.text.apply(lambda x: len(x.strip().split())) >= 10]
    for subred in data['subreddit']:
        if subred not in subdict:
            subdict[subred] = 1
        else: subdict[subred] += 1
    data['text'] = data.text.apply(lambda x: x.lower())
    data['year'] = args.year
    data['month'] = args.month
    data.drop_duplicates(subset=['text'], inplace=True)
    data.dropna(inplace=True)
    data = data[['user','year','month','text','subreddit']]
    new_data = pd.DataFrame(data=None, columns = ['user','year','month','text','subreddit'])
    for k, v in subdict.items():
        if v > 1000:
            if len(data[data['subreddit'].isin([k])]) < 60: continue 
            new_data = pd.concat([new_data,data[data['subreddit'].isin([k])].sample(n=60,random_state=123)])
    data = new_data
    print(len(subdict),len(data)) 

    train_dev, test = train_test_split(data, test_size=0.2, random_state=123, stratify=data[['subreddit']])
    train, dev = train_test_split(train_dev, test_size=0.125, random_state=123, stratify=train_dev[['subreddit']])
    
    train.to_csv('reddit_train_{}.csv'.format(args.year), index=False)
    dev.to_csv('reddit_dev_{}.csv'.format(args.year), index=False)
    test.to_csv('reddit_test_{}.csv'.format(args.year), index=False)

if __name__ == '__main__':
    start_time = time.time()

    main()

    print('---------- {:.1f} minutes ----------'.format((time.time() - start_time) / 60))
    print()