#Import Libraries 
import tweepy as tw
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as mp
   
#Specify the credentials for twitter 
from tweepy import OAuthHandler
consumer_key = 'jz2CHiGujQxAwI4mO0mWhbRGy'
consumer_secret = 'vFkHiUt15rG8DJpt8rogQSFp3I6xIaaylyPUEkGUViGcTVNwYa'
access_token = '842777358046281728-uTAkbeoVNQbvzDEawGqhpWvblEMJJ5s'
access_secret = 'wl8JMJWzNBKKVpZcNxMt0nO5iTqo5VFKNDHuL9fHYLZDW'
auth = OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_secret)
api = tw.API(auth)
