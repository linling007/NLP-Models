# -*- coding: utf-8 -*-
"""
Created on Fri Apr 26 09:14:28 2019

@author: xuyan
"""
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

#做了一个Embedding操作
def make_batch(seq_data):
    input_batch, output_batch, target_batch = [], [], []
    for seq in seq_data:
        for i in range(2):
            seq[i] = seq[i] + 'P' * (n_step - len(seq[i]))
        input = [num_dic[n] for n in seq[0]]
        output = [num_dic[n] for n in ('S' + seq[1])]
        target = [num_dic[n] for n in (seq[1] + 'E')]
        input_batch.append(np.eye(n_class)[input])
        output_batch.append(np.eye(n_class)[output])
        target_batch.append(target) # not one-hot
    # make tensor
    return Variable(torch.Tensor(input_batch)), Variable(torch.Tensor(output_batch)), Variable(torch.LongTensor(target_batch))

dtype = torch.FloatTensor
# S: Symbol that shows starting of decoding input
# E: Symbol that shows starting of decoding output
# P: Symbol that will fill in blank sequence if current batch data size is short than time steps

char_arr = [c for c in 'SEPabcdefghijklmnopqrstuvwxyz']
num_dic = {n: i for i, n in enumerate(char_arr)}

seq_data = [['man', 'women'], ['black', 'white'], ['king', 'queen'], ['girl', 'boy'], ['up', 'down'], ['high', 'low']]

# Seq2Seq Parameter
n_step = 5 #指定字符的宽度
n_hidden = 128 #隐藏层的宽度
n_class = len(num_dic)
batch_size = len(seq_data)
#[数据条数，字符串宽度，每个字符的映射维度]
#input_batch:[6, 5, 29]  output_out:[6, 6, 29]  target_batch:[6,6]  
input_batch, output_batch, target_batch = make_batch(seq_data)
#hidden:[1, 6, 128]
hidden = Variable(torch.zeros(1, batch_size, n_hidden))

#enc_input:[5, 6, 29] dec_input:[6, 6, 29]
enc_input = input_batch.transpose(0, 1) # enc_input: [max_len(=n_step, time step), batch_size, n_class]
dec_input = output_batch.transpose(0, 1) # dec_input: [max_len(=n_step, time step), batch_size, n_class]
#以下我们仅仅更新一步算法的底层操作
#encoder  输入 =>  enc_input:[5, 6, 29] hidden:[1, 6, 128]
#         输出 =>  _:[5, 6, 128]        enc_states:[1, 6, 128]        
enc_cell = nn.RNN(input_size=n_class, hidden_size=n_hidden, dropout=0.5)
_, enc_states = enc_cell(enc_input, hidden)

#decoder  输入 => dec_input:[6,6,29]   enc_states:[1, 6, 128]
#         输出 => outputs:[6, 6, 128]  _:[1, 6, 128]    
dec_cell = nn.RNN(input_size=n_class, hidden_size=n_hidden, dropout=0.5)
outputs, _ = dec_cell(dec_input, enc_states)

#linear 输入 => outputs:[6,6,128]
#       输出 => outputs:[6, 6, 29]   [字符串宽度+1，数据条数，每个字符的映射维度]
fc = nn.Linear(n_hidden, n_class)
#outputs:[6, 6, 29]
outputs = fc(outputs) # model : [max_len+1(=6), batch_size, n_class]
#outputs1:[6,6,29]  [数据条数，字符串宽度+1，每个字符的映射维度]
outputs1 = outputs.transpose(0, 1)

#计算交叉熵损失之和
criterion = nn.CrossEntropyLoss()
loss = 0
for i in range(0, len(target_batch)):
    loss += criterion(outputs1[i], target_batch[i])
#tensor(19.9342, grad_fn=<ThAddBackward>)

#经过若干轮计算，使得loss特别小，从而保存RNN，linear中的各种神经网络参数

#最终选择一个数据进行预测
predict=outputs[3].max(1, keepdim=True)[1]
ll=[char_arr[i] for i in predict]
print(ll)