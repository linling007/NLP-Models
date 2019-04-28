# -*- coding: utf-8 -*-
"""
Created on Fri Apr 26 11:19:09 2019

@author: xuyan
"""

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt


def make_batch(sentences):
    input_batch = [np.eye(n_class)[[word_dict[n] for n in sentences[0].split()]]]
    output_batch = [np.eye(n_class)[[word_dict[n] for n in sentences[1].split()]]]
    target_batch = [[word_dict[n] for n in sentences[2].split()]]
    return Variable(torch.Tensor(input_batch)), Variable(torch.Tensor(output_batch)), Variable(torch.LongTensor(target_batch))

def get_att_weight( dec_output, enc_outputs):  # get attention weight one 'dec_output' with 'enc_outputs'
    n_step = len(enc_outputs)
    attn_scores = Variable(torch.zeros(n_step))  # attn_scores : [n_step]

    for i in range(n_step):
        attn_scores[i] = get_att_score(dec_output, enc_outputs[i])
    return F.softmax(attn_scores).view(1, 1, -1)

def get_att_score( dec_output, enc_output):  # enc_outputs [batch_size, num_directions(=1) * n_hidden]
    attn = nn.Linear(n_hidden, n_hidden)
    score = attn(enc_output)  # score : [batch_size, n_hidden]
    return torch.dot(dec_output.view(-1), score.view(-1))  # inner product make scalar value

dtype = torch.FloatTensor
# S: Symbol that shows starting of decoding input
# E: Symbol that shows starting of decoding output
# P: Symbol that will fill in blank sequence if current batch data size is short than time steps
sentences = ['ich mochte ein bier P', 'S i want a beer', 'i want a beer E']

word_list = " ".join(sentences).split()
word_list = list(set(word_list))
word_dict = {w: i for i, w in enumerate(word_list)}
number_dict = {i: w for i, w in enumerate(word_list)}
n_class = len(word_dict)  # vocab list

n_hidden = 128

#input_batch 指的是sentences[0]  output_batch指的是sentences[1]

#输入  input_batch: [1, 5, 11]  [数据条数，单词个数，映射维度]
#输出  output_batch: [1, 5, 11]
input_batch, output_batch, target_batch = make_batch(sentences)    

#hidden [1, 1, 128]
hidden = Variable(torch.zeros(1, 1, n_hidden))

#输出  enc_inputs: [5, 1, 11]  [单词个数，数据条数，映射维度]
#     dec_inputs: [5, 1, 11]  label
enc_inputs = input_batch.transpose(0, 1)  # enc_inputs: [n_step(=n_step, time step), batch_size, n_class]
dec_inputs = output_batch.transpose(0, 1)  # dec_inputs: [n_step(=n_step, time step), batch_size, n_class]

#输入 enc_inputs  [5, 1, 11]  [单词个数，数据条数，映射维度
#     hidden     [1, 1, 128]
#输出 enc_outputs [5, 1, 128]
#     enc_hidden [1, 1, 128]
#以下是encoder的RNN映射
enc_cell = nn.RNN(input_size=n_class, hidden_size=n_hidden, dropout=0.5)
enc_outputs, enc_hidden = enc_cell(enc_inputs, hidden)

trained_attn = []
hidden = enc_hidden # [1, 1, 128]
n_step = len(dec_inputs)  #5 Attention机制中 dec_inputs可以跟enc_inputs 维度不一样是完全可以的。这个就是改进的点
model = Variable(torch.empty([n_step, 1, n_class])) #[5, 1, 11] [单词个数，数据条数，映射维度]

for i in range(n_step):  # each time step
    #输入 dec_inputs[0].unsqueeze(0): [1, 1, 11]    hidden: [1, 1, 128]
    #输出 dec_output: [1, 1, 128]                   hidden:[1, 1, 128]
    dec_cell = nn.RNN(input_size=n_class, hidden_size=n_hidden, dropout=0.5)
    #过程5
    dec_output, hidden = dec_cell(dec_inputs[i].unsqueeze(0), hidden)
    
    #输入 dec_output: [1, 1, 128]  enc_outputs:[5, 1, 128]
    #输出 attn_weights:[1, 1, 5] 是经过一个softmax的计算的
    #过程4
    attn_weights = get_att_weight(dec_output, enc_outputs)  # attn_weights : [1, 1, n_step]
    #把每一次计算的过程存在trained_attn中
    trained_attn.append(attn_weights.squeeze().data.numpy())
    #过程3
    #输入 attn_weights:[1, 1, 5]
    #     enc_outputs.transpose(0, 1) : [1, 5, 128]   
    #     context: [1,1,128]
    # .bmm 指的是两个矩阵相乘  a.bmm(b)=a*b
    context = attn_weights.bmm(enc_outputs.transpose(0, 1))
    dec_output = dec_output.squeeze(0)  # dec_output : [batch_size(=1), num_directions(=1) * n_hidden]
    context = context.squeeze(1)  # [1, num_directions(=1) * n_hidden] [1,1*128]
    out = nn.Linear(n_hidden * 2, n_class)
    #输入：dec_output:[1, 128]  context:[1,128]
    #输出：resultput[1,256]
    resultput = torch.cat((dec_output, context), 1)
    #经过一个线性回归器
    #输入是 [1,256]  输出是[1,11]
    model[i] = out(resultput)

criterion = nn.CrossEntropyLoss()
#optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

#model [5,11],  target_batch.squeeze(0) [5]
loss = criterion(model.transpose(0, 1).squeeze(0), target_batch.squeeze(0))
#tensor(2.3607, grad_fn=<NllLossBackward>)

# Test
test_batch = [np.eye(n_class)[[word_dict[n] for n in 'SPPPP']]]
test_batch = Variable(torch.Tensor(test_batch))
#predict, trained_attn = model(input_batch, hidden, test_batch)
predict = model.max(2, keepdim=True)[1]
print(sentences[0], '->', [number_dict[n.item()] for n in predict.squeeze()])
   
