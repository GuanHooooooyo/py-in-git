import spacy
import en_qai_sm
from spacy.matcher import PhraseMatcher
import pandas as pd
from spacy.util import minibatch
from ckiptagger import WS,POS,NER
import os
#path = os.path.abspath('D:\CKIPdataset\data')
ws = WS(".\data")
pos = POS(".\data")
ner = NER(".\data")


sentence_list = ["傅達仁今將執行安樂死，卻突然爆出自己20年前遭緯來體育台封殺，他不懂自己哪裡得罪到電視台。",
                "美國參議院針對今天總統布什所提名的勞工部長趙小蘭展開認可聽證會，預料她將會很順利通過參議院支持，成為該國有史以來第一位的華裔女性內閣成員。",
                  "",
                "土地公有政策?？還是土地婆有政策。.",
                "… 你確定嗎… 不要再騙了……",
                "最多容納59,000個人,或5.9萬人,再多就不行了.這是環評的結論.",
                "科長說:1,坪數對人數為1:3。2,可以再增加。"]

word_s = ws(sentence_list,
            sentence_segmentation=True,
            segment_delimiter_set={'?', '？', '!', '！', '。', ',','，', ';', ':', '、'})
word_p = pos(word_s)
word_n = ner(word_s,word_p)

def combine_wandp(w_list, p_list):
    assert len(w_list) == len(p_list)
    for w, p in zip(w_list, p_list):
        print ('{}({})'.format(w, p), end='\u3000')

for i, sentence in enumerate(sentence_list):
    print ("'{}'".format(sentence))
    combine_wandp(word_s[i], word_p[i])
    print ()
    for n in sorted(word_n[i]):
        print (n)
    print('\n')





