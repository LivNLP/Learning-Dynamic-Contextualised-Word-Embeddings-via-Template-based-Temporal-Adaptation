import pandas as pd
from sklearn.model_selection import train_test_split
from collections import defaultdict

def rating_to_label(r):
    if r <= 2:
        return 0
    else:
        return 1


def main():
    
    user_times = defaultdict(list)
    for c in pd.read_json('yelp_academic_dataset_review.json', chunksize=1000, lines=True):
        for user, time in zip(c['user_id'], c['date']):
            user_times[user].append(time)
    
    users = set()
    for user in user_times:
        #t_2004_2021 = [t.year for t in user_times[user] if t.year >= 2004]
        #if len(t_2004_2021) >= 100:
        users.add(user)
    
    data = None
    for c in pd.read_json('yelp_academic_dataset_review.json', chunksize=10000, lines=True):
        c = c[(c['stars'].isin([1, 2, 4, 5])) & (c.date.dt.year >= 2004)]
        if data is None:
            data = c[['user_id', 'date', 'text', 'stars']]
        else:
            data = pd.concat([data, c[['user_id', 'date', 'text', 'stars']]])

    data = data[data.text.apply(lambda x: len(x.strip().split())) >= 10]
    data['text'] = data.text.apply(lambda x: x.lower())
    data['label'] = data['stars'].apply(rating_to_label)
    data['time'] = data.date.dt.date
    data['year'] = data.date.dt.year
    data['month'] = data.date.dt.month
    data['day'] = data.date.dt.weekday
    data.drop_duplicates(subset=['text'], inplace=True)
    data.dropna(inplace=True)
    data.rename(columns={'stars': 'rating', 'user_id': 'user'}, inplace=True)
    
    data.reset_index(inplace=True, drop=True)
    print(data.shape[0])
    assert data.shape[0] == 7671309#745110 #795661

    data = data[['user', 'time', 'year', 'month', 'day', 'text', 'rating', 'label']]

    for i in range(2004,2022):
        temp = data[data.year == i]
        train_dev, test = train_test_split(temp, test_size=0.2, random_state=123, stratify=temp[['rating']])
        train, dev = train_test_split(train_dev, test_size=0.125, random_state=123, stratify=train_dev[['rating']])

        train.to_csv('yelp_train_{}.csv'.format(i), index=False)
        dev.to_csv('yelp_dev_{}.csv'.format(i), index=False)
        test.to_csv('yelp_test_{}.csv'.format(i), index=False)



if __name__ == '__main__':
    main()