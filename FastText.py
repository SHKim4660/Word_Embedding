from gensim.models.fasttext import FastText
import gensim.models.word2vec

path = 'remon_01v_without_19.txt'
sentences = gensim.models.word2vec.Text8Corpus(path)
model = FastText(sentences, min_count=10, vector_size=50, window=5)
print(model)

model.save('fasttext_model')
saved_model = FastText.load('fasttext_model')

# 완성된 임베딩 매트릭스의 크기 확인
print(model.wv.vectors.shape)

# 이순신의 임베딩 벡터를 조회
print(model.wv['레몬'])

# 이순신과 유사성이 있는 데이터 10개 추출
print(model.wv.most_similar(positive=["인공지능"], topn=10))

# 이순신과 이명박의 유사성 수치 검색
print(model.wv.similarity('레몬', '인공지능'))

# 이순신과 원균의 유사성 수치 검색
print(model.wv.similarity('레몬', '사람'))