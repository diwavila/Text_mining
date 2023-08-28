import pandas as pd
import json
from os import listdir
import re

PLAYERS = {
    0: ['Luis Enrique'],  # Luis Enrique
    1: ['Robert'],  # Robert Sánchez
    2: ['Azpilicueta', 'Cesar'],  # César Azpilicueta
    3: ['Eric García', 'Eric Garcia'],  # Eric García
    4: ['Pau Torres'],  # Pau Torres
    5: ['Busquets'],  # Sergio Busquets
    6: ['Llorente'],  # Marcos Llorente
    7: ['Morata'],  # Álvaro Morata
    8: ['Koke'],  # Koke
    9: ['Gavi'],  # Gavi
    10: ['Asensio'],  # Marco Asensio
    11: ['Ferran', 'Torres'],  # Ferran Torres
    12: ['Nico', 'Williams'],  # Nico Williams
    13: ['Raya'],  # David Raya
    14: ['Balde'],  # Alejandro Balde
    15: ['Guillamón'],  # Hugo Guillamón
    16: ['Rodri', 'Rodrigo'],  # Rodrigo
    17: ['Yeremy'],  # Yeremy
    18: ['Alba'],  # Jordi Alba
    19: ['Soler'],  # Carlos Soler
    20: ['Carvajal'],  # Daniel Carvajal
    21: ['Olmo'],  # Dani Olmo
    22: ['Sarabia'],  # Pablo Sarabia
    23: ['Unai', 'Simón', 'Simon'],  # Unai Simón
    24: ['Aymeric', 'Laporte'],  # Aymeric Laporte
    25: ['Ansu', 'Fati'],  # Ansu Fati
    26: ['Pedri'],  # Pedri
    27: ['España']

}


class CSVThings:

    def __init__(self):
        self.df = None

    def create_csv(self):
        path = '/Users/diwavila/Desktop/TDDE16_TextMining/PROJECT/twitter_data/espale/'
        outfile = 'espanya-alemanya.csv'
        files = [f for f in listdir(path) if f.endswith('.json')]
        print(files)

        corpus = []
        for i in files:
            with open(path + i, 'r') as file:
                corpus.append(json.loads(file.read()))

        self.df = pd.DataFrame()
        for i in range(len(corpus)):
            df2 = pd.DataFrame.from_dict(corpus[i])
            self.df = pd.concat([self.df, df2])

        self.df.to_csv(outfile, index=False)

    def load_csv(self, path):

        self.df = pd.read_csv(path)

    def create_players_csv(self, outfile):

        def find_players(x):
            players = []
            for key, values in PLAYERS.items():
                for value in values:
                    if re.search(rf"\b{value}\b", x, re.IGNORECASE):
                        players.append(key)
                        break

            if players:
                return players
            return None

        self.df['players'] = self.df['text'].apply(lambda x: find_players(x))
        lendf = len(self.df)
        self.df = self.df[self.df['players'].notnull()]
        lennot: int = len(self.df)
        print('before:', lendf, 'after:', lennot)
        self.df.to_csv(outfile, index=False)

    def get_players_number_tweets(self):
        lst = [0] * 28
        for index, row in self.df.iterrows():
            if row['players'] != None:
                for player in row['players']:
                    print(player)
                    lst[int(player)] += 1
        return lst

    def df_for_player(self, path, outfile):
        self.load_csv(path)
        #self.get_players_number_tweets()

        for key, values in PLAYERS.items():
            aux = pd.DataFrame()
            for value in values:
                aux2 = self.df[self.df.text.str.contains(value, case=False)]
                aux = pd.concat([aux, aux2], ignore_index=True)
            n = len(aux)
            aux['player'] = [key] * n
            aux.to_excel(outfile + str(key) + '.xlsx', index=False)



def main():
    path_alemanya = '/Users/diwavila/Desktop/TDDE16_TextMining/PROJECT/twitter_data/espanya-alemanya.csv'
    path_costarica = '/Users/diwavila/Desktop/TDDE16_TextMining/PROJECT/twitter_data/espanya_costarica.csv'
    c = CSVThings()

    c.load_csv(path_alemanya)
    c.create_players_csv('espanya_alemanya_cropped.csv')

    c.load_csv(path_costarica)
    c.create_players_csv('espanya_costarica_cropped.csv')


def player_stats():
    c = CSVThings()
    path_alemanya = '/Users/diwavila/Desktop/TDDE16_TextMining/PROJECT/twitter_data/espanya_alemanya_cropped.csv'
    path_costarica = '/Users/diwavila/Desktop/TDDE16_TextMining/PROJECT/twitter_data/espanya_costarica_cropped.csv'

    c.load_csv(path_alemanya)
    alemanya = c.get_players_number_tweets()
    print('alemanya done')

    c.load_csv(path_costarica)
    costarica = c.get_players_number_tweets()
    print('costarica done')

    print('partit alemanya')
    [print(PLAYERS[i], ',', alemanya[i]) for i in range(len(alemanya))]
    print('partit costarica')
    [print(PLAYERS[i], ',', costarica[i]) for i in range(len(costarica))]


def create_lots_of_df():
    path_alemanya = '/Users/diwavila/Desktop/TDDE16_TextMining/PROJECT/twitter_data/espanya-alemanya.csv'
    c = CSVThings()

    c.df_for_player(path_alemanya, 'espanyaALEMANYA')


create_lots_of_df()
