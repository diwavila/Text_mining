import pandas as pd
import json

docs = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 58, 307, 316]

corpus = []
for i in docs:
    with open(f'/Users/diwavila/Desktop/TDDE16_TextMining/PROJECT/twitter_data/train_espcrc/espcrc{i}.json', 'r') as file:
        corpus.append(json.loads(file.read()))

df = pd.DataFrame()
for i in range(len(corpus)):
    df2 = pd.DataFrame.from_dict(corpus[i])
    df = pd.concat([df, df2])

df.to_csv('espanya_costarica.csv', index=False)