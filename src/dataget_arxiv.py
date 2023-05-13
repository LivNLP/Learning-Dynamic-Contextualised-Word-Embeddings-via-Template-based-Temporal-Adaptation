import json
from datetime import datetime
from sklearn.model_selection import train_test_split
import pandas as pd
data_file = 'arxiv-metadata-oai-snapshot.json'

def get_metadata():
    with open(data_file, 'r') as f:
        for line in f:
            yield line


subject_counter = {}
metadata = get_metadata()
for paper in metadata:
    paper_dict = json.loads(paper)
    subjects = paper_dict.get('categories').split()
    for sub in subjects:
        if sub in subject_counter:
            subject_counter[sub] += 1
        else: subject_counter[sub] = 1
subject_list = [k for k,v in subject_counter.items() if v >= 100]
print(subject_list,len(subject_list))

data = {}
data['authors'] = []
data['title'] = []
data['text'] = []
data['subjects'] = []
data['time'] = []
data['year'] = []
data['month'] = []


metadata = get_metadata()
for paper in metadata:
    paper_dict = json.loads(paper)
    subjects = paper_dict.get('categories').split()
    flag = False
    for sub in subjects:
        if sub in subject_list:
            flag = True
    if flag is False: continue
    date = paper_dict.get('versions')[-1]['created']
    date_time = datetime.strptime(date, '%a, %d %b %Y %H:%M:%S GMT')
    # date_time = datetime.strptime(paper_dict.get('update_date'),'%Y-%m-%d')
    data['authors'].append(paper_dict.get('authors'))
    data['title'].append(paper_dict.get('title'))
    data['text'].append(paper_dict.get('abstract'))
    data['subjects'].append(paper_dict.get('categories'))
    data['time'].append(date_time)
    data['year'].append(date_time.year)
    data['month'].append(date_time.month)

data = pd.DataFrame(data)
data['text'] = data.text.apply(lambda x: x.lower())
print(data)
for i in range(2001,2021):
    temp = data[data.year == i]
    print(i, len(temp))
    train_dev, test = train_test_split(temp, test_size=0.2, random_state=123)
    train, dev = train_test_split(train_dev, test_size=0.125, random_state=123)

    train.to_csv('../arxiv_train_{}.csv'.format(i), index=False)
    dev.to_csv('../arxiv_dev_{}.csv'.format(i), index=False)
    test.to_csv('../arxiv_test_{}.csv'.format(i), index=False)