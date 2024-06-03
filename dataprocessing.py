import pandas as pd
import matplotlib.pyplot as plt
import urllib.request
from gensim.models.word2vec import Word2Vec
from konlpy.tag import Okt

targetJson = open('remon_01v_without_19.json','r',encoding='UTF-8')

train_data = pd.read_table(targetJson)

print(len(train_data)) # 리뷰 개수 출력