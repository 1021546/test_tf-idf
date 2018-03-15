# encoding=utf-8
import jieba
import pickle
import math
import time
import numpy

# 變數宣告 --------------------------------------------------------------------------------------------------------
file_qa = 'o1.txt'
file_log = 'log.txt'
context_log = ''

# 整理各個詞 ------------------------------------------------------------------------------------------------------

t0 = time.time()  #計時開始

q_list = []
a_list = []
word_list = []
word_count_list = []
count_line = 0
lines = open(file_qa, 'r', encoding = 'utf8').readlines()
for line in lines:
    # temp = line.split('\t')
    temp = line[:-1].split('\t')
    q_list.append(temp[0])
    a_list.append(temp[1])
    for word in set(temp[0].split(' ')):
        if word not in word_list:
            word_list.append(word)
            word_count_list.append(1)
        else:
            word_count_list[word_list.index(word)] += 1
    count_line += 1
    print('word_list: '+str(count_line)+' ... (OK)') 
# pickle.dump(word_list, open('word_list.pkl', "wb" )) #存檔
# pickle.dump(word_count_list, open('word_count_list.pkl', "wb" )) #存檔
# word_list = pickle.load(open('word_list.pkl', "rb" ))  #載入
# word_count_list = pickle.load(open('word_count_list.pkl', "rb" ))  #載入
t1 = time.time()  #計時結束
print('word_list&word_count_list費時: ' + str(t1-t0) + '秒')  #列印結果

num_q = len(q_list)
idf_list = []
# count_line = 0
for word_count in word_count_list:
    idf_list.append(math.log(num_q/word_count, 10))
    # count_line += 1
    # print('idf_list: '+str(count_line)+' ... (OK)')
# pickle.dump(idf_list, open('idf_list.pkl', "wb" )) #存檔
# idf_list = pickle.load(open('idf_list.pkl', "rb" ))  #載入
t2 = time.time()  #計時結束
print('idf_list費時: ' + str(t2-t1) + '秒')  #列印結果

# count_line = 0
# q_word_id_list = []
# q_tf_idf_list = []
q_norm_list = []
q_word_id_tf_idf_list = []
for q in q_list:
    # word_id_list = []
    # tf_idf_list = []
    word_id_tf_idf_dic = {}
    q_words = q.split(' ')
    q_len = len(q_words)
    for word in set(q_words):
        word_index = word_list.index(word)
        # word_id_list.append(word_index)
        # tf_idf_list.append((q.count(word)/q_len) * (idf_list[word_index]))
        word_id_tf_idf_dic[word_index] = (q.count(word)/q_len) * (idf_list[word_index])
    # q_word_id_list.append(word_id_list)
    # q_tf_idf_list.append(tf_idf_list)
    q_word_id_tf_idf_list.append(word_id_tf_idf_dic)
    q_norm_list.append(numpy.linalg.norm(list(word_id_tf_idf_dic.values())))
    # count_line += 1
    # print('q_tf_idf_list: '+str(count_line)+' ... (OK)')
# pickle.dump(q_word_id_list, open('q_word_id_list.pkl', "wb" )) #存檔
# pickle.dump(q_tf_idf_list, open('q_tf_idf_list.pkl', "wb" )) #存檔
# pickle.dump(q_norm_list, open('q_norm_list.pkl', "wb" )) #存檔
# q_tf_idf_list = pickle.load(open('q_tf_idf_list.pkl', "rb" ))  #載入
# q_norm_list = pickle.load(open('q_norm_list.pkl', "rb" ))  #載入
t3 = time.time()  #計時結束
print('q_tf_idf_list費時: ' + str(t3-t2) + '秒')  #列印結果

# 輸出 ------------------------------------------------------------------------------------------------------------
# file_w = open(file_log, 'w', encoding = 'utf8')
# file_w.write(context_log)
# file_w.close()

# pickle.dump((q_list, a_list, word_list, idf_list, q_word_id_list, q_tf_idf_list, q_norm_list), open('vsm.pkl', "wb" )) #存檔
pickle.dump((q_list, a_list, word_list, idf_list, q_word_id_tf_idf_list, q_norm_list), open('vsm.pkl', "wb" )) #存檔
# (q_list, a_list, word_list, idf_list, q_word_id_list, q_tf_idf_list, q_norm_list) = pickle.load(open('vsm.pkl', "rb" ))  #載入


