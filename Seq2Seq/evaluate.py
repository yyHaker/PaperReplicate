# -*- coding: utf-8 -*-
"""Evaluation utils."""
import torch
from torch.autograd import Variable
from data_utils import get_minibatch, get_autoencode_minibatch
from decode import BeamSearchDecoder
import numpy as np
from bleu import get_bleu


def decode_minibatch(
    config,
    model,
    input_lines_src,
    input_lines_trg,
    output_lines_trg_gold,
    use_cuda=False
):
    """Decode a minibatch greedily.
    批量的decode every word in a sentence.
    :param config:
    :param model:
    :param input_lines_src: 从encoder输入的句子，[batch, seq]
    :param input_lines_trg: Variable, decoder 输入的部分，[batch, 1]
    :param output_lines_trg_gold:
    :param use_cuda: if use cuda
    :return:
            "input_lines_trg": [batch, max_trg_length]
    """
    for i in range(config['data']['max_trg_length']):
        decoder_logit = model(input_lines_src, input_lines_trg)
        word_probs = model.decode(decoder_logit)  # [batch, trg_seq_len, trg_vocab_size]
        decoder_argmax = word_probs.data.cpu().numpy().argmax(axis=-1)
        next_preds = Variable(
            torch.from_numpy(decoder_argmax[:, -1])
        )
        if use_cuda:
            next_preds = next_preds.cuda()

        input_lines_trg = torch.cat((input_lines_trg, next_preds.unsqueeze(1)), 1)
    return input_lines_trg


def model_perplexity(
    model, src, src_test, trg,
    trg_test, config, loss_criterion,
    src_valid=None, trg_valid=None, verbose=False,
):
    """Compute model perplexity."""
    # Get source minibatch
    losses = []
    for j in range(0, len(src_test['data']) // 100, config['data']['batch_size']):
        input_lines_src, output_lines_src, lens_src, mask_src = get_minibatch(
            src_test['data'], src['word2id'], j, config['data']['batch_size'],
            config['data']['max_src_length'], add_start=True, add_end=True
        )
        input_lines_src = Variable(input_lines_src.data, volatile=True)
        output_lines_src = Variable(input_lines_src.data, volatile=True)
        mask_src = Variable(mask_src.data, volatile=True)

        # Get target minibatch
        input_lines_trg_gold, output_lines_trg_gold, lens_src, mask_src = (
            get_minibatch(
                trg_test['data'], trg['word2id'], j,
                config['data']['batch_size'], config['data']['max_trg_length'],
                add_start=True, add_end=True
            )
        )
        input_lines_trg_gold = Variable(input_lines_trg_gold.data, volatile=True)
        output_lines_trg_gold = Variable(output_lines_trg_gold.data, volatile=True)
        mask_src = Variable(mask_src.data, volatile=True)

        decoder_logit = model(input_lines_src, input_lines_trg_gold)

        loss = loss_criterion(
            decoder_logit.contiguous().view(-1, decoder_logit.size(2)),
            output_lines_trg_gold.view(-1)
        )

        losses.append(loss.data[0])

    return np.exp(np.mean(losses))


def evaluate_model(
    model, src, src_test, trg,
    trg_test, config, src_valid=None, trg_valid=None,
    verbose=True, metric='bleu', use_cuda=False
):
    """Evaluate model.
    :param model: the model object
    :param src:
    :param src_test:
    :param trg:
    :param trg_test:
    :param config: the config object
    :param src_valid:
    :param trg_valid:
    :param verbose:
    :param metric:
    :param use_cuda:
    :return:
    """
    preds = []
    ground_truths = []
    for j in range(0, len(src_test['data']), config['data']['batch_size']):
        # Get source minibatch
        input_lines_src, output_lines_src, lens_src, _ = get_minibatch(
            src_test['data'], src['word2id'], j, config['data']['batch_size'],
            config['data']['max_src_length'], add_start=True, add_end=True,
            use_cuda=use_cuda)

        # Get target minibatch
        input_lines_trg_gold, output_lines_trg_gold, lens_src, _ = (
            get_minibatch(
                trg_test['data'], trg['word2id'], j,
                config['data']['batch_size'], config['data']['max_trg_length'],
                add_start=True, add_end=True, use_cuda=use_cuda
            ))

        # Initialize target with <s> for every sentence
        input_lines_trg = Variable(torch.LongTensor(
            [
                [trg['word2id']['<s>']]
                for i in range(input_lines_src.size(0))
            ]
        ))
        if use_cuda:
            input_lines_trg = input_lines_trg.cuda()

        # print("input_lines_src: ", input_lines_src.size(), "input_lines_trg: ", input_lines_trg.size())
        # input_lines_src: [80, 49],   "input_lines_trg: " [80, 1]
        # Decode a minibatch greedily __TODO__ add beam search decoding
        input_lines_trg = decode_minibatch(
            config, model, input_lines_src,
            input_lines_trg, output_lines_trg_gold,
            use_cuda=use_cuda
        )
        # save gpu memory(in vain)
        input_lines_src = input_lines_src.data.cpu().numpy()
        del input_lines_src
        output_lines_src = output_lines_src.data.cpu().numpy()
        input_lines_trg_gold = input_lines_trg_gold.data.cpu().numpy()
        del input_lines_trg_gold

        # Copy minibatch outputs to cpu and convert ids to words
        input_lines_trg = input_lines_trg.data.cpu().numpy()
        input_lines_trg = [
            [trg['id2word'][x] for x in line]
            for line in input_lines_trg
        ]

        # Do the same for gold sentences
        output_lines_trg_gold = output_lines_trg_gold.data.cpu().numpy()
        output_lines_trg_gold = [
            [trg['id2word'][x] for x in line]
            for line in output_lines_trg_gold
        ]
        print("input_lines_trg: ", input_lines_trg[0])
        print("the length  of a sent", len(input_lines_trg[0]))
        # Process outputs
        for sentence_pred, sentence_real, sentence_real_src in zip(
            input_lines_trg,
            output_lines_trg_gold,
            output_lines_src
        ):
            # 去除开始和结束符， 构造完整的句子sentence, 以便计算bleu值
            if '<s>' in sentence_pred:
                index = sentence_pred.index('<s>')
                sentence_pred = sentence_pred[index+1:]
            if '</s>' in sentence_pred:
                index = sentence_pred.index('</s>')
                sentence_pred = sentence_pred[:index]
            preds.append(sentence_pred)

            if '<s>' in sentence_real:
                index = sentence_real.index('<s>')
                sentence_real = sentence_real[index+1:]
            if '</s>' in sentence_real:
                index = sentence_real.index('</s>')
                sentence_real = sentence_real[: index]
            ground_truths.append(sentence_real)

    print("call the get_bleu method to calc bleu score.....")
    print("preds: ", preds[0])
    print("ground_truths: ", ground_truths[0])
    return get_bleu(preds, ground_truths)


def evaluate_autoencode_model(
    model, src, src_test,
    config, src_valid=None,
    verbose=True, metric='bleu'
):
    """Evaluate model."""
    preds = []
    ground_truths = []
    for j in xrange(0, len(src_test['data']), config['data']['batch_size']):

        print('Decoding batch : %d out of %d ' % (j, len(src_test['data'])))
        input_lines_src, lens_src, mask_src = get_autoencode_minibatch(
            src_test['data'], src['word2id'], j, config['data']['batch_size'],
            config['data']['max_src_length'], add_start=True, add_end=True
        )

        input_lines_trg = Variable(torch.LongTensor(
            [
                [src['word2id']['<s>']]
                for i in range(input_lines_src.size(0))
            ]
        )).cuda()

        for i in range(config['data']['max_src_length']):

            decoder_logit = model(input_lines_src, input_lines_trg)
            word_probs = model.decode(decoder_logit)
            decoder_argmax = word_probs.data.cpu().numpy().argmax(axis=-1)
            next_preds = Variable(
                torch.from_numpy(decoder_argmax[:, -1])
            ).cuda()

            input_lines_trg = torch.cat(
                (input_lines_trg, next_preds.unsqueeze(1)),
                1
            )

        input_lines_trg = input_lines_trg.data.cpu().numpy()

        input_lines_trg = [
            [src['id2word'][x] for x in line]
            for line in input_lines_trg
        ]

        output_lines_trg_gold = input_lines_src.data.cpu().numpy()
        output_lines_trg_gold = [
            [src['id2word'][x] for x in line]
            for line in output_lines_trg_gold
        ]

        for sentence_pred, sentence_real in zip(
            input_lines_trg,
            output_lines_trg_gold,
        ):
            if '</s>' in sentence_pred:
                index = sentence_pred.index('</s>')
            else:
                index = len(sentence_pred)
            preds.append(sentence_pred[:index + 1])

            if verbose:
                print(' '.join(sentence_pred[:index + 1]))

            if '</s>' in sentence_real:
                index = sentence_real.index('</s>')
            else:
                index = len(sentence_real)
            if verbose:
                print(' '.join(sentence_real[:index + 1]))
            if verbose:
                print('--------------------------------------')
            ground_truths.append(sentence_real[:index + 1])
    return get_bleu(preds, ground_truths)


def evaluate_model_beam_search(model, src, src_test, trg,
    trg_test, config, beam_size=1, use_cuda=False):
    """
    evaluate the model use beam search.
    :param model:
    :param src:
    :param src_test:
    :param trg:
    :param trg_test:
    :param config:
    :param beam_size:
    :param use_cuda:
    :return:
    """
    # the test word dict use src's dict
    src_test['word2id'] = src['word2id']
    src_test['id2word'] = src['id2word']

    trg_test['word2id'] = trg['word2id']
    trg_test['id2word'] = trg['id2word']

    decoder = BeamSearchDecoder(config, model.state_dict(),
                                src_test, trg_test, beam_size=beam_size, use_cuda=use_cuda)
    bleu_score = decoder.translate()
    return bleu_score