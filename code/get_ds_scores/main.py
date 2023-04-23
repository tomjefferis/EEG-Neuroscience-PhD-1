import os
import pandas as pd

if __name__ == '__main__':

    data = os.path.join(os.getcwd(), 'data')
    fnames = os.listdir(data)

    all_data = []
    for p, each_file in enumerate(fnames):
        with open(os.path.join(data, each_file)) as f:
            dataset = f.readlines()
            trial = 0
            for line in dataset:
                items = line.split('\t')
                if 'Gray' in items:
                    discomfort_score = float(items[4])
                    all_data.append([p, trial, discomfort_score])
                    trial+=1

    data = pd.DataFrame(all_data, columns=['participant', 'trial', 'discomfort_score'])

    scores = []
    for trial, data in data.groupby('trial'):
        avg_discomfort = data['discomfort_score'].mean()
        scores.append([trial, avg_discomfort])

    scores = pd.DataFrame(scores, columns=['trial', 'ds_score'])
    scores.to_csv('scores.csv', index=False)