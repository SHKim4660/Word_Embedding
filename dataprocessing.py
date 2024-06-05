import pandas as pd
import matplotlib.pyplot as plt
from gensim.models.word2vec import Word2Vec
from konlpy.tag import Okt


targetJson = open('remon_01v_without_19.json','r',encoding='UTF-8')

train_data = pd.read_json(targetJson)

# print(train_data.isnull().values.any()) # 결측치 존재여부 확인 -> False

train_data_list = []

for i in range(len(train_data['conversations'])):
    for j in range(len(train_data['conversations'][i])):
        train_data_replaced = str(train_data['conversations'][i][j]['value']).replace("[^ㄱ-ㅎㅏ-ㅣ가-힣 ]","") # 정규식 이용 한글제외 문자 제거
        train_data['conversations'][i][j]['value'] = train_data_replaced
        print("0")
        

# 불용어 정의
stopwords = ['의','가','이','은','들','는','좀','잘','걍','과','도','를','으로','자','에','와','한','하다']

# 형태소 분석기 OKT를 사용한 토큰화 작업 (다소 시간 소요)
okt = Okt()

tokenized_data = []
for i in range(len(train_data['conversations'])):
    for j in range(len(train_data['conversations'][i])):
        sentence = train_data['conversations'][i][j]['value']
        tokenized_sentence = okt.morphs(sentence, stem=True) # 토큰화
        stopwords_removed_sentence = [word for word in tokenized_sentence if not word in stopwords] # 불용어 제거
        tokenized_data.append(stopwords_removed_sentence)
        print("1")

# 리뷰 길이 분포 확인
print('문장의 최대 길이 :',max(len(review) for review in tokenized_data))
print('문장의 평균 길이 :',sum(map(len, tokenized_data))/len(tokenized_data))
plt.hist([len(review) for review in tokenized_data], bins=50)
plt.xlabel('length of samples')
plt.ylabel('number of samples')
plt.show()

from gensim.models import Word2Vec

print("2")

model = Word2Vec(sentences = tokenized_data, vector_size = 100, window = 5, min_count = 5, workers = 4, sg = 0)

print("3")

# 완성된 임베딩 매트릭스의 크기 확인
model.wv.vectors.shape

print(model.wv.most_similar("바나나"))

print(model.wv.most_similar("안녕"))