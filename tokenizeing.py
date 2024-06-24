import pandas as pd
import matplotlib.pyplot as plt
from konlpy.tag import Okt


def data_tokenizeing(path):
    cnt = 0

    targetJson = open(path,'r',encoding='UTF-8')

    train_data = pd.read_json(targetJson)

    if not train_data.isnull().values.any(): # 결측치 존재여부 확인
        pass
    else: print("결측치 존재.. 제거함"); 
         

    train_data_list = []

    for i in range(len(train_data['conversations'])):
        for j in range(len(train_data['conversations'][i])):
            train_data_replaced = str(train_data['conversations'][i][j]['value']).replace("[^ㄱ-ㅎㅏ-ㅣ가-힣 ]","") # 정규식 이용 한글제외 문자 제거
            train_data['conversations'][i][j]['value'] = train_data_replaced
            cnt += 1
            print(f"making list{cnt}...")
            

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
            print("data tkenizing...")

    # 길이 분포 확인
    print(f'Toknizing된 데이터 수{cnt}')
    print('문장의 최대 길이 :',max(len(review) for review in tokenized_data))
    print('문장의 평균 길이 :',sum(map(len, tokenized_data))/len(tokenized_data))
    plt.hist([len(review) for review in tokenized_data], bins=50)
    plt.xlabel('length of samples')
    plt.ylabel('number of samples')
    plt.show()

    return tokenized_data
