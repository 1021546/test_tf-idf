# encoding=utf-8
import jieba
import pickle
import math
from scipy import spatial
import numpy
import time

# 變數宣告 --------------------------------------------------------------------------------------------------------
# file_qa = 'o1.txt'
# file_log = 'log.txt'
# context_log = ''

# 載入vsm模型
# (q_list, a_list, word_list, idf_list, q_word_id_list, q_tf_idf_list, q_norm_list) = pickle.load(open('vsm.pkl', "rb" ))  #載入
(q_list, a_list, word_list, idf_list, q_word_id_tf_idf_list, q_norm_list) = pickle.load(open('vsm.pkl', "rb" ))  #載入

# 載入斷詞詞典
jieba.set_dictionary('dict_.txt')
jieba.load_userdict('教育部辭典_163709.dic')

# 測試 ------------------------------------------------------------------------------------------------------------

t0 = time.time()  #計時開始
test_sentence = "對天龍人來說宜蘭4南部還４東部"
temp = jieba.cut(test_sentence, cut_all=False)
test_sentence = " ".join(temp)  # 將測試句斷詞
test_sentence = test_sentence.replace('   ', ' ')
# test_sentence = "對 天龍 人 來說 宜蘭 4 南部 還 ４ 東部"
print(test_sentence)
t1 = time.time()  #計時結束
print('斷詞費時: ' + str(t1-t0) + '秒')  #列印結果

# test_word_id_list = []
# test_tf_idf_list = []
test_word_id_tf_idf_dic = {}
test_words = test_sentence.split(' ')
test_len = len(test_words)
for word in set(test_words):
    if word in word_list:
        word_index = word_list.index(word)
        # test_word_id_list.append(word_index)
        # test_tf_idf_list.append((test_words.count(word)/test_len) * (idf_list[word_index]))
        test_word_id_tf_idf_dic[word_index] = (test_words.count(word)/test_len) * (idf_list[word_index])
# print(str(test_tf_idf_list))
t2 = time.time()  #計時結束
print('tf-idf費時: ' + str(t2-t1) + '秒')  #列印結果

similar_list = []
# test_norm = numpy.linalg.norm(test_tf_idf_list)
test_norm = numpy.linalg.norm(list(test_word_id_tf_idf_dic.values()))

def sim():
    # for q_i, q_v in enumerate(q_word_id_list):
    for q_i, q_v in enumerate(q_word_id_tf_idf_list):
        # result = 0
        # for t_i, t_v in enumerate(test_word_id_list):
        #     # if q_v.count(t_v) == 1:
        #     if t_v in q_v:
        #         result += test_tf_idf_list[t_i]*q_tf_idf_list[q_i][q_v.index(t_v)]
        # result /= (test_norm*q_norm_list[q_i])
        # similar_list.append(result)
        result = 0
        for i in (test_word_id_tf_idf_dic.keys() & q_v):
            result += test_word_id_tf_idf_dic[i]*q_v[i]
        result /= (test_norm*q_norm_list[q_i])
        similar_list.append(result)
sim()

t3 = time.time()  #計時結束
print('相似度費時: ' + str(t3-t2) + '秒')  #列印結果
print(similar_list[numpy.argmax(similar_list)])
print(numpy.argmax(similar_list))
print(q_list[numpy.argmax(similar_list)])
print(a_list[numpy.argmax(similar_list)])

# 輸出 ------------------------------------------------------------------------------------------------------------
# file_w = open(file_log, 'w', encoding = 'utf8')
# file_w.write(context_log)
# file_w.close()
