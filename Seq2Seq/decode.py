# -*- coding: utf-8 -
"""
use Beam Search to decoding.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from model import Seq2Seq
from data_utils import read_nmt_data, get_minibatch, read_config
from beam_search import Beam
from bleu import get_bleu


class BeamSearchDecoder(object):
    """Beam Search decoder."""
    def __init__(
        self,
        config,
        model_weights,
        src,
        trg,
        beam_size=1,
        use_cuda=False
    ):
        """Initialize model."""
        self.config = config
        self.model_weights = model_weights
        self.beam_size = beam_size

        self.use_cuda = use_cuda

        self.src = src
        self.trg = trg
        self.src_dict = src['word2id']
        self.tgt_dict = trg['word2id']
        self._load_model()

    def _load_model(self):
        print('Loading pretrained model')
        if self.config['model']['seq2seq'] == 'vanilla':
            print('Loading Seq2Seq Vanilla model')

            self.model = Seq2Seq(
                src_emb_dim=self.config['model']['dim_word_src'],
                trg_emb_dim=self.config['model']['dim_word_trg'],
                src_vocab_size=len(self.src_dict),
                trg_vocab_size=len(self.tgt_dict),
                src_hidden_dim=self.config['model']['dim_src'],
                trg_hidden_dim=self.config['model']['dim_trg'],
                pad_token_src=self.src_dict['<pad>'],
                pad_token_trg=self.tgt_dict['<pad>'],
                use_cuda=self.use_cuda,
                batch_size=self.config['data']['batch_size'],
                bidirectional=self.config['model']['bidirectional'],
                nlayers=self.config['model']['n_layers_src'],
                nlayers_trg=self.config['model']['n_layers_trg'],
                dropout=0.,
            )
            if self.use_cuda:
                self.model = self.model.cuda()

        self.model.load_state_dict(
            self.model_weights
        )

    def get_hidden_representation(self, input):
        """Get hidden representation for a sentence."""
        src_emb = self.model.src_embedding(input)
        h0_encoder, c0_encoder = self.model.get_state(src_emb)
        src_h, (src_h_t, src_c_t) = self.model.encoder(
            src_emb, (h0_encoder, c0_encoder)
        )

        if self.model.bidirectional:
            h_t = torch.cat((src_h_t[-1], src_h_t[-2]), 1)
            c_t = torch.cat((src_c_t[-1], src_c_t[-2]), 1)
        else:
            h_t = src_h_t[-1]
            c_t = src_c_t[-1]

        return src_h, (h_t, c_t)

    def get_init_state_decoder(self, input):
        """Get init state for decoder."""
        decoder_init_state = nn.Tanh()(self.model.encoder2decoder(input))
        return decoder_init_state

    def decode_batch(self, idx):
        """
        Decode a minibatch.
        :param idx:
        :return:
        """
        # Get source minibatch
        input_lines_src, output_lines_src, lens_src, mask_src = get_minibatch(
            self.src['data'], self.src_dict, idx,
            self.config['data']['batch_size'],
            self.config['data']['max_src_length'], add_start=True, add_end=True,
            use_cuda=self.use_cuda
        )

        beam_size = self.beam_size

        #  (1) run the encoder on the src

        context_h, (context_h_t, context_c_t) = self.get_hidden_representation(
            input_lines_src
        )

        context_h = context_h.transpose(0, 1)  # Make things sequence first.

        #  (3) run the decoder to generate sentences, using beam search

        batch_size = context_h.size(1)

        # Expand tensors for each beam.
        context = Variable(context_h.data.repeat(1, beam_size, 1))
        dec_states = [
            Variable(context_h_t.data.repeat(1, beam_size, 1)),
            Variable(context_c_t.data.repeat(1, beam_size, 1))
        ]

        beam = [
            Beam(beam_size, self.tgt_dict, cuda=self.use_cuda)
            for k in range(batch_size)
        ]

        dec_out = self.get_init_state_decoder(dec_states[0].squeeze(0))
        dec_states[0] = dec_out

        batch_idx = list(range(batch_size))
        remaining_sents = batch_size

        for i in range(self.config['data']['max_trg_length']):

            input = torch.stack(
                [b.get_current_state() for b in beam if not b.done]
            ).t().contiguous().view(1, -1)

            trg_emb = self.model.trg_embedding(Variable(input).transpose(1, 0))
            trg_h, (trg_h_t, trg_c_t) = self.model.decoder(
                trg_emb,
                (dec_states[0].squeeze(0), dec_states[1].squeeze(0)),
                # context
            )

            dec_states = (trg_h_t.unsqueeze(0), trg_c_t.unsqueeze(0))

            dec_out = trg_h_t.squeeze(1)
            out = F.softmax(self.model.decoder2vocab(dec_out)).unsqueeze(0)

            word_lk = out.view(
                beam_size,
                remaining_sents,
                -1
            ).transpose(0, 1).contiguous()

            active = []
            for b in range(batch_size):
                if beam[b].done:
                    continue

                idx = batch_idx[b]
                if not beam[b].advance(word_lk.data[idx]):
                    active += [b]

                for dec_state in dec_states:  # iterate over h, c
                    # layers x beam*sent x dim
                    # print(dec_state.size())  # [1, 1, 80, 1000]
                    sent_states = dec_state.view(
                        -1, beam_size, remaining_sents, dec_state.size(3)
                    )[:, :, idx]
                    sent_states.data.copy_(
                        sent_states.data.index_select(
                            1,
                            beam[b].get_current_origin()
                        )
                    )

            if not active:
                break

            # in this section, the sentences that are still active are
            # compacted so that the decoder is not run on completed sentences
            if self.use_cuda:
                active_idx = torch.cuda.LongTensor([batch_idx[k] for k in active])
            else:
                active_idx = torch.LongTensor([batch_idx[k] for k in active])
            batch_idx = {beam: idx for idx, beam in enumerate(active)}

            def update_active(t):
                # select only the remaining active sentences
                view = t.data.view(
                    -1, remaining_sents,
                    self.model.decoder.hidden_size
                )
                new_size = list(t.size())
                new_size[-2] = new_size[-2] * len(active_idx) \
                    // remaining_sents
                return Variable(view.index_select(
                    1, active_idx
                ).view(*new_size))

            dec_states = (
                update_active(dec_states[0]),
                update_active(dec_states[1])
            )
            dec_out = update_active(dec_out)
            context = update_active(context)

            remaining_sents = len(active)

        #  (4) package everything up

        allHyp, allScores = [], []
        n_best = 1

        for b in range(batch_size):
            scores, ks = beam[b].sort_best()

            allScores += [scores[:n_best]]
            hyps = zip(*[beam[b].get_hyp(k) for k in ks[:n_best]])
            allHyp += [hyps]

        return allHyp, allScores

    def translate(self):
        """Translate the whole dataset."""
        trg_preds = []
        trg_gold = []
        for j in range(
            0, len(self.src['data']),
            self.config['data']['batch_size']
        ):
            """Decode a single minibatch."""
            print('Decoding %d out of %d ' % (j, len(self.src['data'])))
            hypotheses, scores = self.decode_batch(j)
            all_hyp_inds = [[x[0] for x in hyp] for hyp in hypotheses]
            all_preds = [
                ' '.join([self.trg['id2word'][x] for x in hyp])
                for hyp in all_hyp_inds
            ]

            # Get target minibatch
            input_lines_trg_gold, output_lines_trg_gold, lens_src, mask_src = (
                get_minibatch(
                    self.trg['data'], self.tgt_dict, j,
                    self.config['data']['batch_size'],
                    self.config['data']['max_trg_length'],
                    add_start=True, add_end=True, use_cuda=self.use_cuda
                )
            )

            output_lines_trg_gold = output_lines_trg_gold.data.cpu().numpy()
            all_gold_inds = [[x for x in hyp] for hyp in output_lines_trg_gold]
            all_gold = [
                ' '.join([self.trg['id2word'][x] for x in hyp])
                for hyp in all_gold_inds
            ]

            trg_preds += all_preds
            trg_gold += all_gold
        print("investigate some preds and golds.....")
        print("trg_preds: ", trg_preds[0])
        print("trg_gold: ", trg_gold[0])
        bleu_score = get_bleu(trg_preds, trg_gold)

        # print('BLEU : %.5f ' % (bleu_score))
        return bleu_score
