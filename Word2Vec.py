import pandas as pd
import matplotlib.pyplot as plt
from gensim.models.word2vec import Word2Vec
from konlpy.tag import Okt
import tokenizeing

tokenized_data = tokenizeing.data_tokenizeing('remon_01v_without_19.json')

from gensim.models import Word2Vec

print("training Model...")

model = Word2Vec(sentences = tokenized_data, min_count = 10, vector_size= 50, window = 5)

print("Done")

model.wv.save_word2vec_format('eng_w2v')
model.save('w2v_model')

print("Save Model..")

model = Word2Vec.load('w2v_model')

# 완성된 임베딩 매트릭스의 크기 확인
print(model.wv.vectors.shape)

# 이순신의 임베딩 벡터를 조회
print(model.wv['레몬'])

# 이순신과 유사성이 있는 데이터 10개 추출
print(model.wv.most_similar(positive=["레몬"], topn=10))

# 이순신과 이명박의 유사성 수치 검색
print(model.wv.similarity('레몬', '인공'))

# 이순신과 원균의 유사성 수치 검색
print(model.wv.similarity('레몬', '사람'))
