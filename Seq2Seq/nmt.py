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
torch.backends.cudnn.enabled = False

parser = argparse.ArgumentParser()
# 参数配置文件路径， 必须输入该参数
parser.add_argument("--config", help="path to config",
                    default="config_en_vanilla_seq2seq_wmt14.json")
args = parser.parse_args()

config_file_path = args.config
config = read_config(config_file_path)
experiment_name = hyperparam_string(config)
save_dir = config['data']['save_dir']  # 模型保存路径
load_dir = config['data']['load_dir']  # 模型保存路径

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

# cost a lot memory
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
max_src_length = config['data']['max_src_length']  # 源句子最大长度
max_trg_length = config['data']['max_trg_length']  # 目标句子最大长度
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
        src_hidden_dim=config['model']['dim_src'],
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

if load_dir:
    model.load_state_dict(torch.load(open(load_dir)))

# __TODO__ Make this more flexible for other learning methods.
if config['training']['optimizer'] == 'adam':
    lr = config['training']['lrate']
    optimizer = optim.Adam(model.parameters(), lr=lr)
elif config['training']['optimizer'] == 'adadelta':
    optimizer = optim.Adadelta(model.parameters())
elif config['training']['optimizer'] == 'sgd':
    lr = config['training']['lrate']
    optimizer = optim.SGD(model.parameters(), lr=lr)
else:
    raise NotImplementedError("Learning method not recommend for task")

# training.....
for i in range(1000):
    losses = []
    for j in range(0, len(src['data']), batch_size):
        # get batch data
        input_lines_src, _, lens_src, mask_src = get_minibatch(
            src['data'], src['word2id'], j,
            batch_size, max_src_length, add_start=True, add_end=True,
            use_cuda=True
        )
        input_lines_trg, output_lines_trg, lens_trg, mask_trg = get_minibatch(
            trg['data'], trg['word2id'], j,
            batch_size, max_trg_length, add_start=True, add_end=True,
            use_cuda=True
        )

        decoder_logit = model(input_lines_src, input_lines_trg)
        optimizer.zero_grad()

        # loss(x, class) = weights[class] * (-x[class] + log(\sum_j exp(x[j])))
        loss = loss_criterion(
            decoder_logit.contiguous().view(-1, trg_vocab_size),
            output_lines_trg.view(-1)
        )
        losses.append(loss.data[0])
        loss.backward()
        optimizer.step()

        # every 'monitor_loss' step to monitor
        if j % config['management']['monitor_loss'] == 0:
            logging.info('Epoch : %d Minibatch : %d Loss : %.5f' % (
                i, j, np.mean(losses)))
            losses = []

        if (config['management']['print_samples'] and
                        j % config['management']['print_samples'] == 0):
            word_probs = model.decode(
                decoder_logit
            ).data.cpu().numpy().argmax(axis=-1)

            output_lines_trg = output_lines_trg.data.cpu().numpy()

            # sample several sentence
            for sentence_pred, sentence_real in zip(
                word_probs[:5], output_lines_trg[:5]
            ):
                sentence_pred = [trg['id2word'][x] for x in sentence_pred]
                sentence_real = [trg['id2word'][x] for x in sentence_real]

                if '</s>' in sentence_real:
                    index = sentence_real.index('</s>')
                    sentence_real = sentence_real[:index]
                    sentence_pred = sentence_pred[:index]

                logging.info('Predicted : %s ' % (' '.join(sentence_pred)))
                logging.info('-----------------------------------------------')
                logging.info('Real : %s ' % (' '.join(sentence_real)))
                logging.info('===============================================')

            if j % config['management']['checkpoint_freq'] == 0:
                logging.info('Evaluating model ...')
                bleu = evaluate_model(
                    model, src, src_test, trg,
                    trg_test, config, verbose=False,
                    metric='bleu',
                    use_cuda=use_cuda
                )

                logging.info('Epoch : %d Minibatch : %d : BLEU : %.5f ' % (i, j, bleu))

                logging.info('Saving model ...')

                torch.save(
                    model.state_dict(),
                    open(os.path.join(
                        save_dir,
                        experiment_name + '__epoch_%d__minibatch_%d' % (i, j) + '.model'), 'wb'
                    )
                )
        # every epoch calculate bleu
        bleu = evaluate_model(
            model, src, src_test, trg,
            trg_test, config, verbose=False,
            metric='bleu',
            use_cuda=use_cuda
        )

        logging.info('Epoch : %d : BLEU : %.5f ' % (i, bleu))

        torch.save(
            model.state_dict(),
            open(os.path.join(
                save_dir,
                experiment_name + '__epoch_%d' % (i) + '.model'), 'wb'
            )
        )















