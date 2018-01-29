# -*- coding: utf-8 -*-
"""Sequence to Sequence models"""
import torch
import torch.nn as nn
from torch.autograd import Variable
import math
import torch.nn.functional as F
import numpy as np


class Seq2Seq(nn.Module):
    """Container module with an encoder, decoder, embeddings."""
    def __init__(
            self,
            src_emb_dim,  # 源句子中单词向量维度
            trg_emb_dim,  # 目标句子中单词向量维度
            src_vocab_size,
            trg_vocab_size,
            src_hidden_dim,
            trg_hidden_dim,
            pad_token_src,
            pad_token_trg,
            use_cuda=False,         # 是否使用cuda
            batch_size=64,
            bidirectional=True,
            nlayers=2,        # 源RNN的层数
            nlayers_trg=1,  # 目标RNN的层数
            dropout=0.       # RNN层与层之间的dropout
    ):
        """Initialize model."""
        super(Seq2Seq, self).__init__()
        self.src_emb_dim = src_emb_dim
        self.trg_emb_dim = trg_emb_dim
        self.src_vocab_size = src_vocab_size
        self.trg_vocab_size = trg_vocab_size
        self.src_hidden_dim = src_hidden_dim
        self.trg_hidden_dim = trg_hidden_dim
        # If given, pads the output with zeros whenever it encounters the index
        self.pad_token_src = pad_token_src
        self.pad_token_trg = pad_token_trg
        self.use_cuda = use_cuda
        self.batch_size = batch_size
        self.bidirectional = bidirectional
        self.nlayers = nlayers
        self.nlayers_trg = nlayers_trg
        self.dropout = dropout

        # self.src_hidden_dim才是encoder RNN中的hidden_size
        self.src_hidden_dim = src_hidden_dim // 2 if self.bidirectional else src_hidden_dim

        self.num_directions = 2 if bidirectional else 1

        self.src_embedding = nn.Embedding(
            self.src_vocab_size,
            self.src_emb_dim,
            self.pad_token_src
        )

        self.trg_embedding = nn.Embedding(
            self.trg_vocab_size,
            self.trg_emb_dim,
            self.pad_token_trg
        )

        # encoder and decoder use LSTM
        self.encoder = nn.LSTM(
            input_size=self.src_emb_dim,
            hidden_size=self.src_hidden_dim,
            num_layers=self.nlayers,  # number of recurrent layers
            bidirectional=self.bidirectional,
            batch_first=True,  # If True, then the input and output tensors are provided as (batch, seq, feature)
            dropout=self.dropout
        )

        self.decoder = nn.LSTM(
            input_size=self.trg_emb_dim,
            hidden_size=self.trg_hidden_dim,
            num_layers=self.nlayers_trg,
            dropout=self.dropout,
            batch_first=True  # If True, then the input and output tensors are provided as (batch, seq, feature)
        )

        self.encoder2decoder = nn.Linear(
            in_features=self.src_hidden_dim * self.num_directions,
            out_features=self.trg_hidden_dim
        )

        self.decoder2vocab = nn.Linear(trg_hidden_dim, trg_vocab_size)

        if self.use_cuda:
            self.encoder2decoder = nn.Linear(
                in_features=self.src_hidden_dim * self.num_directions,
                out_features=self.trg_hidden_dim
            ).cuda()
            self.decoder2vocab = nn.Linear(trg_hidden_dim, trg_vocab_size).cuda()

        # initialize weights
        self.init_weights()

    def init_weights(self):
        """Initialize Embedding weights and Linear bias."""
        initrange = 0.1
        self.src_embedding.weight.data.uniform_(-initrange, initrange)
        self.trg_embedding.weight.data.uniform_(-initrange, initrange)

        self.encoder2decoder.bias.data.fill_(0)
        self.decoder2vocab.bias.data.fill_(0)

    def get_state(self, input):
        """Get Encoder cell states and hidden states.
        :param input: If batch_first, then [seq_len, batch]
        :return: （h0, c0）(num_layers * num_directions, batch, hidden_size)
        """
        batch_size = input.size(0) if self.encoder.batch_first else input.size(1)
        h0_encoder = Variable(torch.zeros(
            self.encoder.num_layers * self.num_directions,
            batch_size,
            self.src_hidden_dim
        ))
        c0_encoder = Variable(torch.zeros(
            self.encoder.num_layers * self.num_directions,
            batch_size,
            self.src_hidden_dim
        ))
        if self.use_cuda:
            return h0_encoder.cuda(), c0_encoder.cuda()
        else:
            return h0_encoder, c0_encoder

    def forward(self, input_src, input_trg, ctx_mask=None, trg_mask=None):
        """
        Propagate input through the network.
        :param input_src: [batch, seq_len_src]
        :param input_trg: [batch, seq_len_trg]
        :param ctx_mask:
        :param trg_mask:
        :return:
        """
        # 得到源句子的向量[batch, seq_len_src, src_emb_dim]
        src_emb = self.src_embedding(input_src)
        # 得到目标句子的向量[batch, seq_len_trg, trg_emb_dim]
        trg_emb = self.trg_embedding(input_trg)

        # 得到初始LSTM的输入变量h0、c0
        self.h0_encoder, self.c0_encoder = self.get_state(input_src)

        # Encode the input
        src_h, (src_h_t, src_c_t) = self.encoder(
            src_emb, (self.h0_encoder, self.c0_encoder)
        )

        # src_h: [seq_len_src, batch, self.src_hidden_dim * num_directions]
        # src_h_t: [num_layers * num_directions, batch, self.src_hidden_dim]
        # src_c_t: [num_layers * num_directions, batch, self.src_hidden_dim]
        if self.bidirectional:
            h_t = torch.cat((src_h_t[-1], src_h_t[-2]), 1)  # 在列上拼接[batch, self.src_hidden_dim*2]
            c_t = torch.cat((src_c_t[-1], src_c_t[-2]), 1)  # 在列上拼接[batch, self.src_hidden_dim*2]
        else:
            h_t = src_h_t[-1]   # [batch, self.src_hidden_dim]
            c_t = src_c_t[-1]   # [batch, self.src_hidden_dim]

        # h_t  -> [batch, trg_hidden_dim]
        decoder_init_state = nn.Tanh()(self.encoder2decoder(h_t))

        # decode
        # LSTM  input: (batch, seq_len, self.trg_emb_dim),
        # (h0, c0): (num_layers(默认为1)*num_directions(默认为1), batch, self.trg_hidden_dim)
        # print("----decoder LSTM dim inspect---------")
        # print("trg_emb: ", trg_emb.size(), "to be h0: ", decoder_init_state, "to be c0: ", c_t)
        trg_h, (_, _) = self.decoder(
            trg_emb,   # 80x49x500
            (
                decoder_init_state.view(    # 1x80x1000
                    self.decoder.num_layers,
                    decoder_init_state.size(0),
                    decoder_init_state.size(1)
                ),
                c_t.view(                         # 1x80x1000
                    self.decoder.num_layers,
                    c_t.size(0),
                    c_t.size(1)  # 这里的"hidden_size"?怎么理解?
                )                    # 需要encoder的hidden_size 和decoder的hidden_size一样
            )
        )

        # trg_h : [batch, trg_seq_len, trg_hidden_dim*nlayers_trg]
        trg_h_reshape = trg_h.contiguous().view(
            trg_h.size(0) * trg_h.size(1),
            trg_h.size(2)
        )
        # [trg_seq_len*batch, trg_hidden_dim*nlayers_trg] -> [trg_seq_len*batch, trg_vocab_size]
        decoder_logit = self.decoder2vocab(trg_h_reshape)
        decoder_logit = decoder_logit.view(
            trg_h.size(0),
            trg_h.size(1),
            decoder_logit.size(1)
        )  # [batch, trg_seq_len, trg_vocab_size]
        return decoder_logit

    def decode(self, logits):
        """
        Return probability distribution over words.
        :param logits: [batch, trg_seq_len, trg_vocab_size] or [*, trg_vocab_size]
        :return: 'word_probs' [batch, trg_seq_len, trg_vocab_size] or [*, trg_vocab_size]
        """
        logits_reshape = logits.view(-1, self.trg_vocab_size)
        word_probs = F.softmax(logits_reshape)
        word_probs = word_probs.view(
            logits.size()[0], logits.size()[1], logits.size()[2]
        )
        return word_probs



