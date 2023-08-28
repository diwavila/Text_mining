import tweepy
from tweepy import OAuthHandler
import json
from datetime import datetime
from sys import exit

import time
import random

consumer_key = 'svtDn8fvQKR6dBvSLGpqnQllB'
consumer_secret = 'DXAcZGXkNP3ZGs3xDFp2UQEyvfxC4D3jZJKDLVBKcMmzRCUAfB'
access_token = '1386626109014429698-EyKLD1ir9Z6QVC0imdJv2VD7dQnx7D'
access_secret = 'FaPBqXKxgbBSfTYNxn7annedTsXNk4LcPEIvXVgO2MJxM'
print('trying to connect')
auth = OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_secret)
api = tweepy.API(auth, wait_on_rate_limit=True)  # this is that if you bus one
#  of the wait limits and you don't know about it you weren't tracking it, this
#  will wait for you. It will just stop the program from running and wait until
#  you're available. If you can only do 50 things at every one hour, it will do
#  50 and wait until your api limit is
print('connected')
search_list = ['qatar2022', 'mundial2022', 'qatarworldcup2022', 'fifaworldcup']
search_list_esp = ['españa', 'seleccion española', 'selección española', 'vamosespana', 'vamosespaña']
players_list = ['unai', 'simón', 'simon', 'busquets', 'luis enrique', 'azpilicueta', 'olmo', 'laporte',
                'aymeric', 'alba', 'rodrigo', 'rodri', 'pedri', 'gavi', 'ferran', 'torres', 'asensio', 'ausencio','eric garcia', 'eric garcía', 'eric', 'morata'
                'álvaro', 'balde', 'ansu', 'fati', 'nico', 'williams', 'llorente','marcos', 'koke', 'Guillamón', 'guillamon',
                'carvajal', 'pau torres', 'robert sanchez', 'robert', 'raya', 'españa', 'seleccion española', 'selección española', 'soler', 'carlos', 'sarabia', 'yeremy']


class MyStreamListener(tweepy.Stream):
    def __init__(self):
        self.j = 0
        self.i = 0
        super(MyStreamListener, self).__init__(consumer_key, consumer_secret, access_token, access_secret)
        self.data = []

    def on_status(self, status):

        if datetime.now().strftime("%H:%M:%S") > '22:15:00':
            with open(f'espjap{self.i}.json', 'a') as final:
                json.dump(self.data, final)
            exit()
            return False

        if datetime.now().strftime("%H:%M:%S") > '19:45:00':

            if not hasattr(status, 'retweeted_status'):
                print(self.i, self.j)
                #print(status)
                id = status.id_str
                #text = None
                text = (
                    status.extended_tweet['full_text']
                    if hasattr(status, 'extended_tweet')
                    else status.text
                )

                reply = (
                    status.in_reply_to_status_id_str
                    if status.in_reply_to_status_id_str is not None
                    else None
                )

                created_at = status._json['created_at']
                quoted = (
                    status.quoted_status_id_str
                    if hasattr(status, 'quoted_status_id_str')
                    else None
                )

                self.data.append({'id': id,
                                  'created_at': created_at,
                                  'text': text,
                                  'reply_id': reply,
                                  'quote_id': quoted})

                self.j += 1

            if self.j == 500:
                with open(f'espmar{self.i}.json', 'a') as final:
                    json.dump(self.data, final)

                self.i += 1
                self.j = 0
                self.data = []
            #print(self.i, self.j)

            return True

        else:
            return True

    def on_error(self, status):
        if status == 420:
            print('Enhance Your Calm; The App Is Being Rate Limited For Making Too Many Requests')
            return True
        else:
            print('Error {}n'.format(status))
            return True


def streamtweets():
    # listener = MyStreamListener()
    #stream = tweepy.Stream(auth=api.auth, listener=listener)

    stream = MyStreamListener()
    """myStream = tweepy.Stream(consumer_key=consumer_key,
                             consumer_secret=consumer_secret, access_token=access_token, access_token_secret=access_secret)"""
    stream.filter(track=players_list, languages=['es'])

streamtweets()


