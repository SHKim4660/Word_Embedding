from gensim.models.fasttext import FastText
import gensim.models.word2vec

path = 'remon_01v_without_19.json'
sentences = gensim.models.word2vec.Text8Corpus(path)
model = FastText(sentences, min_count=10, size=50, window=5)
print(model)

model.save('fasttext_model')
saved_model = FastText.load('fasttext_model')
word_vector = saved_model['이순신']
print(word_vector)

print(model.most_similar(positive=["이순신"], topn=10))
print(model.similarity('이순신', '이명박'))
print(model.similarity('이순신', '원균'))

model.most_similar(positive=['대한민국', '베이징'], negative=['서울'])
print(model.similar_by_word('카카오톡'))