# -*- coding: utf-8 -*-
"""Main script to run things"""
import sys
from data_utils import read_nmt_data, get_minibatch, read_config, hyperparam_string
from model import Seq2Seq
from evaluate import evaluate_model
import numpy as np
import logging
import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

# 设置全局cudnn不可用
torch.backends.cudnn.enable = False

parser = argparse.ArgumentParser()
# 参数配置文件路径， 必须输入该参数
parser.add_argument("--config", help="path to config",
                    default="config_en_vanilla_seq2seq_wmt14.json")
args = parser.parse_args()

config_file_path = args.config
config = read_config(config_file_path)
experiment_name = hyperparam_string(config)
save_dir = config['data']['save_dir']
load_dir = config['data']['load_dir']

# 创建log file
log_file_name = 'log/%s' % experiment_name
with open(log_file_name, 'w') as f:
    pass

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='log/%s' % experiment_name,
    filemode='w'
)

# define a new Handler to log to console as well
console = logging.StreamHandler()
# optional, set the logging level
console.setLevel(logging.INFO)
# set a format which is the same for console use
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
# tell the handler to use this format
console.setFormatter(formatter)
# add the handler to the root logger
logging.getLogger('').addHandler(console)

print("Reading data.....")

src, trg = read_nmt_data(
    src=config['data']['src'],
    config=config,
    trg=config['data']['trg']
)

src_test, trg_test = read_nmt_data(
    src=config['data']['test_src'],
    config=config,
    trg=config['data']['test_trg']
)

use_cuda = config['model']['use_cuda']
batch_size = config['data']['batch_size']
max_length = config['data']['max_src_length']  # 源句子最大长度
src_vocab_size = len(src['word2id'])
trg_vocab_size = len(trg['word2id'])

logging.info('Model Parameters : ')
logging.info('Task : %s ' % (config['data']['task']))                    # 任务类别
logging.info('Model : %s ' % (config['model']['seq2seq']))           # 模型类别
logging.info('Source Language : %s ' % (config['model']['src_lang']))
logging.info('Target Language : %s ' % (config['model']['trg_lang']))
logging.info('Source Word Embedding Dim  : %s' % (config['model']['dim_word_src']))
logging.info('Target Word Embedding Dim  : %s' % (config['model']['dim_word_trg']))
logging.info('Source RNN Hidden Dim  : %s' % (config['model']['dim_src']))
logging.info('Target RNN Hidden Dim  : %s' % (config['model']['dim_trg']))
logging.info('Source RNN Depth : %d ' % (config['model']['n_layers_src']))
logging.info('Target RNN Depth : %d ' % config['model']['n_layers_trg'])
logging.info('Source RNN Bidirectional  : %s' % (config['model']['bidirectional']))
logging.info('Batch Size : %d ' % (config['model']['batch_size']))
logging.info('Optimizer : %s ' % (config['training']['optimizer']))
logging.info('Learning Rate : %f ' % (config['training']['lrate']))
logging.info('Use cuda: %s' % (config['model']['use_cuda']))

logging.info('Found %d words in src ' % src_vocab_size)
logging.info('Found %d words in trg ' % trg_vocab_size)

# 不计算<pad>的loss
weight_mask = torch.ones(trg_vocab_size)
if use_cuda:
    weight_mask = weight_mask.cuda()
weight_mask[trg['word2id']['<pad>']] = 0
loss_criterion = nn.CrossEntropyLoss(weight=weight_mask)
if use_cuda:
    loss_criterion = loss_criterion.cuda()

if config['model']['seq2seq'] == 'vanilla':
    model = Seq2Seq(
        src_emb_dim=config['model']['dim_word_src'],
        trg_emb_dim=config['model']['dim_word_trg'],
        src_vocab_size=src_vocab_size,
        trg_vocab_size=trg_vocab_size,
        src_hidden_dim=config['model']['dim'],
        trg_hidden_dim=config['model']['dim_trg'],
        pad_token_src=src['word2id']['<pad>'],
        pad_token_trg=trg['word2id']['<pad>'],
        use_cuda=use_cuda,
        batch_size=batch_size,
        bidirectional=config['model']['bidirectional'],
        nlayers=config['model']['n_layers_src'],
        nlayers_trg=config['model']['n_layers_trg'],
        dropout=0.
    )
    if use_cuda:
        model = model.cuda()







