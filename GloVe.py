# python 3.8로 실행 glove pakeage가 너무오래되서 최신버전 지원안함 
# py -3.8 -m pip install glove-python-binary

from glove import Corpus, Glove
import tokenizeing

token = tokenizeing.data_tokenizeing('remon_01v_without_19.json')

# corpus 생성
corpus = Corpus()
corpus.fit(token, window=20)

# model
glove = Glove(no_components=128, learning_rate=0.01)     # 0.05
glove.fit(corpus.matrix, epochs=50, no_threads=4, verbose=False)    # Wall time: 8min 32s
glove.add_dictionary(corpus.dictionary)

# save
glove.save('/glove_w20_epoch50.model')

# load glove
glove_model = Glove.load('/glove_w20_epoch50.model')

# word dict 생성
word_dict = {}
for word in  glove_model.dictionary.keys():
    word_dict[word] = glove_model.word_vectors[glove_model.dictionary[word]]
print('[Success !] Lengh of word dict... : ', len(word_dict))

# # save word_dict
# with open('/glove_word_dict_128.pickle', 'wb') as f:
#     pickle.dump(word_dict, f)
# print('[Success !] Save word dict!...')

