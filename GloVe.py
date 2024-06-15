import numpy as np
from glove import Corpus, Glove

# 샘플 문장들
sentences = [
    "The cute cat eats on the sofa",
    "The quick brown fox jumps over the lazy dog",
    "The dog barks at the cat",
    "Cats and dogs are cute animals"
]

# 단어 단위로 토큰화
tokenized_sentences = [sentence.lower().split() for sentence in sentences]

# GloVe 코퍼스 생성
corpus = Corpus()
corpus.fit(tokenized_sentences, window=5)

# GloVe 모델 학습
glove = Glove(no_components=100, learning_rate=0.05)
glove.fit(corpus.matrix, epochs=30, no_threads=4, verbose=True)
glove.add_dictionary(corpus.dictionary)

# 학습된 벡터의 크기 확인
print(glove.word_vectors.shape)

# 단어 유사도 확인
print(glove.most_similar("cat"))
print(glove.most_similar("dog"))
