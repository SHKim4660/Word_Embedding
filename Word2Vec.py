import pandas as pd
import matplotlib.pyplot as plt
from gensim.models.word2vec import Word2Vec
from konlpy.tag import Okt
import tokenizeing

tokenized_data = tokenizeing.data_tokenizeing('remon_01v_without_19.json')

from gensim.models import Word2Vec

print("training Model...")

model = Word2Vec(sentences = tokenized_data, vector_size = 100, window = 5, min_count = 5, workers = 4, sg = 0)

print("Done")

model.wv.save_word2vec_format('eng_w2v')

print("Save Model..")