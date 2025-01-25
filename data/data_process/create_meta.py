import os

import pandas as pd


def ASVspoof2019_LA():
    root = 'ASVspoof2019/LA/ASVspoof2019_LA_cm_protocols'
    files = [
        'ASVspoof2019.LA.cm.train.trl.txt',
        'ASVspoof2019.LA.cm.dev.trl.txt',
        'ASVspoof2019.LA.cm.eval.trl.txt'
    ]
    dir = 'data/metadata'
    dirs_new = [
        'ASVspoof2019_LA_train',
        'ASVspoof2019_LA_dev',
        'ASVspoof2019_LA_eval'
    ]
    for i, file in enumerate(files):
        with open(os.path.join(root, file), 'r') as f:
            lines = f.readlines()
        df = pd.DataFrame()
        for line in lines:
            line = line.strip().split(' ')
            speaker = line[0]
            file = line[1]
            label = line[-1]
            attack = line[-2]
            file_path = f'data/audio/{dirs_new[i]}/flac/{file}.flac'
            df = df.append({'file': file_path,
                            'label': label,
                            'speaker': speaker,
                            'attack': attack},
                           ignore_index=True)
        df.to_csv(os.path.join(dir, f'{dirs_new[i]}.csv'), index=False)


if __name__ == '__main__':
    ASVspoof2019_LA()
