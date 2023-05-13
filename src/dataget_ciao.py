from datetime import datetime
from sklearn.model_selection import train_test_split
import pandas as pd
data_file = 'data/ciao/rating.txt'

data = {}
data['author'] = []
data['text'] = []
data['time'] = []
data['year'] = []
data['month'] = []

with open(data_file) as f:
    lines = f.readlines()

non_cnt = 0
cnt = 0
for line in lines:
    line = line.split("::::")
    author = line[0]
    date_time = None
    text = None
    for index in range(5,len(line)):
        try:
            date_time = datetime.strptime(line[index], '%d.%m.%Y').date()
            text = line[index + 1]
            break
        except:
            continue
    # if author in line[1]:
    #     time = line[6]
    #     text = line[7]
    # else: 
    #     time = line[5]
    #     text = line[6]
    if text is None or date_time is None:
        non_cnt += 1
        continue
    print(date_time, cnt, non_cnt)
    cnt += 1
    # date_time = datetime.strptime(time, '%d.%m.%Y').date()
    data['author'].append(author)
    data['text'].append(text)
    data['time'].append(date_time)
    data['year'].append(date_time.year)
    data['month'].append(date_time.month)
    


data = pd.DataFrame(data)
data['text'] = data.text.apply(lambda x: x.lower())
print(data)
for i in range(2000,2012):
    temp = data[data.year == i]
    print(i, len(temp))
    train_dev, test = train_test_split(temp, test_size=0.2, random_state=123)
    train, dev = train_test_split(train_dev, test_size=0.125, random_state=123)

    train.to_csv('data/ciao_train_{}.csv'.format(i), index=False)
    dev.to_csv('data/ciao_dev_{}.csv'.format(i), index=False)
    test.to_csv('data/ciao_test_{}.csv'.format(i), index=False)