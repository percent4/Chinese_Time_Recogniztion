# -*- coding: utf-8 -*-
# time: 2019-08-09 16:47
# place: Zhichunlu Beijing

import kashgari
from kashgari.corpus import DataReader
from kashgari.embeddings import BERTEmbedding
from kashgari.tasks.labeling import BiLSTM_CRF_Model

train_x, train_y = DataReader().read_conll_format_file('./data/time.train')
valid_x, valid_y = DataReader().read_conll_format_file('./data/time.dev')
test_x, test_y = DataReader().read_conll_format_file('./data/time.test')

bert_embedding = BERTEmbedding('chinese_wwm_ext_L-12_H-768_A-12',
                               task=kashgari.LABELING,
                               sequence_length=128)

model = BiLSTM_CRF_Model(bert_embedding)
model.fit(train_x, train_y, valid_x, valid_y, batch_size=16, epochs=10)

model.save('time_ner.h5')

model.evaluate(test_x, test_y)