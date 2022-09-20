from collections import defaultdict
from typing import List, Optional, Tuple, Dict

import torch

from wenet.transformer.ctc import CTC
from wenet.emformer.encoder import Emformer
from wenet.emformer.decoder import TransformerDecoder
from wenet.transformer.decoder import BiTransformerDecoder
from wenet.transformer.label_smoothing_loss import LabelSmoothingLoss
from wenet.utils.common import IGNORE_ID, LOG_EPS, add_sos_eos, log_add
from wenet.utils.mask import make_pad_mask
from wenet.bin.stream import Stream, stack_states, unstack_states
from torch.nn.utils.rnn import pad_sequence
from wenet.utils.common import (remove_duplicates_and_blank, prepare_loss_inputs, 
                                initializer, reverse_pad_list)
import torch.nn.functional as F
from wenet.transducer.joint_network import JointNetwork
from wenet.transducer.predictor import Predictor
from warprnnt_pytorch import RNNTLoss


class StreamASRModel(torch.nn.Module):
    """CTC-attention hybrid Encoder-Decoder model"""
    def __init__(
        self,
        vocab_size: int,
        encoder: Emformer,
        decoder: TransformerDecoder,
        ctc: CTC,
        predictor: Predictor,
        joint_network: JointNetwork,
        ctc_weight: float = 0.5,
        ignore_id: int = IGNORE_ID,
        reverse_weight: float = 0.0,
        lsm_weight: float = 0.0,
        length_normalized_loss: bool = False,
    ):
        super().__init__()
        # note that eos is the same as sos (equivalent ID)
        self.sos = vocab_size - 1
        self.eos = vocab_size - 1
        self.blank_id = 0
        self.vocab_size = vocab_size
        self.ignore_id = ignore_id
        self.ctc_weight = ctc_weight
        self.reverse_weight = reverse_weight

        self.encoder = encoder
        self.decoder = decoder
        self.ctc = ctc

        self.criterion_att = LabelSmoothingLoss(
            size=vocab_size,
            padding_idx=ignore_id,
            smoothing=lsm_weight,
            normalize_length=length_normalized_loss,
        )
        self.predictor = predictor
        self.joint_network = joint_network
        self.transducer_loss = RNNTLoss(
            blank=self.blank_id,
            reduction="mean",
            fastemit_lambda=0.0,
        )

    def forward(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
        warmup: float = 1.0,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor],
               Optional[torch.Tensor]]:
        """Frontend + Encoder + Decoder + Calc loss

        Args:
            speech: (Batch, Length, ...)
            speech_lengths: (Batch, )
            text: (Batch, Length)
            text_lengths: (Batch,)
        """
        assert text_lengths.dim() == 1, text_lengths.shape
        # Check that batch_size is unified
        assert (speech.shape[0] == speech_lengths.shape[0] == text.shape[0] ==
                text_lengths.shape[0]), (speech.shape, speech_lengths.shape,
                                         text.shape, text_lengths.shape)
        encoder_chunk_out, encoder_out_lens = self.encoder(speech, speech_lengths, warmup=warmup)
        encoder_mask = ~make_pad_mask(encoder_out_lens, encoder_chunk_out.size(1)).unsqueeze(1) 

        # ctc
        loss_ctc = self.ctc(encoder_chunk_out, encoder_out_lens, text, text_lengths)

        
        # 2a. Attention-decoder branch
        if self.ctc_weight != 1.0:
            loss_att = self._calc_att_loss(encoder_chunk_out, encoder_mask,
                                                    text, text_lengths)
        else:
            loss_att = None

        # transducer
        predict_ys_in_pad, target, target_len = prepare_loss_inputs(text, encoder_mask)
        self.predictor.set_device(encoder_chunk_out.device)
        predictor_out = self.predictor(predict_ys_in_pad)
        h_enc = encoder_chunk_out.unsqueeze(2)
        h_dec = predictor_out.unsqueeze(1)
        joint_out = self.joint_network(h_enc, h_dec)
        target = target.to(dtype=torch.int32)
        encoder_out_lens = encoder_out_lens.to(dtype=torch.int32)
        loss_trans = self.transducer_loss(joint_out, target, encoder_out_lens, target_len)

        loss = loss_ctc * self.ctc_weight + loss_att + loss_trans

        return loss, loss_att, loss_ctc, loss_trans

    def _calc_att_loss(
        self,
        encoder_out: torch.Tensor,
        encoder_mask: torch.Tensor,
        ys_pad: torch.Tensor,
        ys_pad_lens: torch.Tensor,
    ) -> torch.Tensor:
        ys_in_pad, ys_out_pad = add_sos_eos(ys_pad, self.sos, self.eos,
                                            self.ignore_id)
        ys_in_lens = ys_pad_lens + 1

        # reverse the seq, used for right to left decoder
        r_ys_pad = reverse_pad_list(ys_pad, ys_pad_lens, float(self.ignore_id))
        r_ys_in_pad, r_ys_out_pad = add_sos_eos(r_ys_pad, self.sos, self.eos,
                                                self.ignore_id)
        # 1. Forward decoder
        decoder_out, r_decoder_out, _ = self.decoder(encoder_out, encoder_mask,
                                                     ys_in_pad, ys_in_lens,
                                                     r_ys_in_pad,
                                                     self.reverse_weight)
        # 2. Compute attention loss
        loss_att = self.criterion_att(decoder_out, ys_out_pad)
        r_loss_att = torch.tensor(0.0)
        if self.reverse_weight > 0.0:
            r_loss_att = self.criterion_att(r_decoder_out, r_ys_out_pad)
        loss_att = loss_att * (
            1 - self.reverse_weight) + r_loss_att * self.reverse_weight

        return loss_att

    def ctc_greedy_search(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        device: torch.device,
        params: Dict,
    ) -> List[List[int]]:
        """ Apply CTC greedy search

        Args:
            speech (torch.Tensor): (batch, max_len, feat_dim)
            speech_length (torch.Tensor): (batch, )
        Returns:
            List[List[int]]: best path result
        """
        assert speech.shape[0] == speech_lengths.shape[0] == 1
        speech_lengths += params['right_context_length']
        speech = torch.nn.functional.pad(
            speech,
            pad=(0, 0, 0, params['right_context_length']),
            value=LOG_EPS,
        )
        encoder_chunk_out, encoder_out_lens = self.encoder(x=speech, x_lens=speech_lengths)

        ctc_probs = self.ctc.log_softmax(encoder_chunk_out)  # (B, maxlen, vocab_size)
        topk_prob, topk_index = ctc_probs.topk(1, dim=2)  # (B, maxlen, 1)
        topk_index = topk_index[0].view(-1)  # (B, maxlen)
        result = topk_index.tolist()

        return [remove_duplicates_and_blank(result)]


    def streaming_ctc_greedy_search(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        device: torch.device,
        params: Dict,
    ) -> List[List[int]]:
        """ Apply CTC greedy search

        Args:
            speech (torch.Tensor): (batch, max_len, feat_dim)
            speech_length (torch.Tensor): (batch, )
        Returns:
            List[List[int]]: best path result
        """
        assert speech.shape[0] == speech_lengths.shape[0] == 1

        stream = Stream(params=params, device=device, LOG_EPS=LOG_EPS)
        stream.set_feature(speech[0])
        
        result = []
        tail_length = 3 * params['subsampling_factor'] + params['right_context_length'] + 3
        while not stream.done:
            num_processed_frames = torch.tensor(stream.num_processed_frames, device=device).unsqueeze(0)
            feature = stream.get_feature_chunk()
            feature = feature.unsqueeze(0) # (1, chunk, 80)
            feature_len = torch.tensor(feature.size(1), device=device).unsqueeze(0)

            if feature.size(1) < tail_length:
                pad_length = tail_length - feature.size(1)
                feature_len += pad_length
                feature = F.pad(feature, (0, 0, 0, pad_length), mode="constant", value=LOG_EPS)

            states = stack_states([stream.states])

            encoder_chunk_out, encoder_out_lens, states = self.encoder.infer(
                x=feature,
                x_lens=feature_len,
                num_processed_frames=num_processed_frames,
                states=states,
            )
            ctc_probs = self.ctc.log_softmax(encoder_chunk_out)  # (B, maxlen, vocab_size)
            topk_prob, topk_index = ctc_probs.topk(1, dim=2)  # (B, maxlen, 1)
            topk_index = topk_index[0].view(-1)  # (B, maxlen)
            result += topk_index.tolist()
            stream.states = unstack_states(states)[0]

        hyps = [remove_duplicates_and_blank(result)]
        return hyps

    def streaming_ctc_prefix_beam_search(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        beam_size: int,
        device: torch.device,
        params: Dict,
    ) -> List[int]:
        hyps, _ = self._ctc_prefix_beam_search(speech, speech_lengths,
                                        beam_size, device, params)
        return hyps[0]


    def _ctc_prefix_beam_search(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        beam_size: int,
        device: torch.device,
        params: Dict,
    ) -> Tuple[List[List[int]], torch.Tensor]:

        assert speech.shape[0] == speech_lengths.shape[0] == 1
        stream = Stream(params=params, device=device, LOG_EPS=LOG_EPS)
        stream.set_feature(speech[0])
        
        tail_length = 3 * params['subsampling_factor'] + params['right_context_length'] + 3

        t = 0
        ctc_probs = torch.empty((0, params['output_dim']), device=device)
        encoder_out = torch.empty((1, 0, params['encoder_dim']), device=device)
        cur_hyps = [(tuple(), (0.0, -float('inf')))]
        while not stream.done:
            num_processed_frames = torch.tensor(stream.num_processed_frames, device=device).unsqueeze(0)
            feature = stream.get_feature_chunk()
            feature = feature.unsqueeze(0) # (1, chunk, 80)
            feature_len = torch.tensor(feature.size(1), device=device).unsqueeze(0)

            if feature.size(1) < tail_length:
                pad_length = tail_length - feature.size(1)
                feature_len += pad_length
                feature = torch.nn.functional.pad(feature, (0, 0, 0, pad_length),
                        mode="constant", value=LOG_EPS)

            states = stack_states([stream.states])

            encoder_chunk_out, encoder_out_lens, states = self.encoder.infer(
                x=feature,
                x_lens=feature_len,
                num_processed_frames=num_processed_frames,
                states=states,
            )
            stream.states = unstack_states(states)[0]

            probs = self.ctc.log_softmax(encoder_chunk_out).squeeze(0)  # (maxlen, vocab_size)
            ctc_probs = torch.cat([ctc_probs, probs])
            encoder_out = torch.cat([encoder_out, encoder_chunk_out], dim=1)

            while ctc_probs.shape[0] > t:
                logp = ctc_probs[t]  # (vocab_size,)
                next_hyps = defaultdict(lambda: (-float('inf'), -float('inf')))
                top_k_logp, top_k_index = logp.topk(beam_size)  # (beam_size,)
                for s in top_k_index:
                    s = s.item()
                    ps = logp[s].item()
                    for prefix, (pb, pnb) in cur_hyps:
                        last = prefix[-1] if len(prefix) > 0 else None
                        if s == 0:  # blank
                            n_pb, n_pnb = next_hyps[prefix]
                            n_pb = log_add([n_pb, pb + ps, pnb + ps])
                            next_hyps[prefix] = (n_pb, n_pnb)
                        elif s == last:
                            #  Update *ss -> *s;
                            n_pb, n_pnb = next_hyps[prefix]
                            n_pnb = log_add([n_pnb, pnb + ps])
                            next_hyps[prefix] = (n_pb, n_pnb)
                            # Update *s-s -> *ss, - is for blank
                            n_prefix = prefix + (s, )
                            n_pb, n_pnb = next_hyps[n_prefix]
                            n_pnb = log_add([n_pnb, pb + ps])
                            next_hyps[n_prefix] = (n_pb, n_pnb)
                        else:
                            n_prefix = prefix + (s, )
                            n_pb, n_pnb = next_hyps[n_prefix]
                            n_pnb = log_add([n_pnb, pb + ps, pnb + ps])
                            next_hyps[n_prefix] = (n_pb, n_pnb)
                # 2.2 Second beam prune
                next_hyps = sorted(next_hyps.items(),
                                key=lambda x: log_add(list(x[1])),
                                reverse=True)
                cur_hyps = next_hyps[:beam_size]

                t += 1
        hyps = [(y[0], log_add([y[1][0], y[1][1]])) for y in cur_hyps]
        return hyps, encoder_out


    def attention_rescoring(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        beam_size: int,
        device: torch.device,
        params: Dict,
        ctc_weight: float = 0.0,
    ) -> List[int]:
        """ Apply attention rescoring decoding, CTC prefix beam search
            is applied first to get nbest, then we resoring the nbest on
            attention decoder with corresponding encoder out

        Args:
            speech (torch.Tensor): (batch, max_len, feat_dim)
            speech_length (torch.Tensor): (batch, )
            beam_size (int): beam size for beam search
            decoding_chunk_size (int): decoding chunk for dynamic chunk
                trained model.
                <0: for decoding, use full chunk.
                >0: for decoding, use fixed chunk size as set.
                0: used for training, it's prohibited here
            simulate_streaming (bool): whether do encoder forward in a
                streaming fashion
            reverse_weight (float): right to left decoder weight
            ctc_weight (float): ctc score weight

        Returns:
            List[int]: Attention rescoring result
        """
        assert speech.shape[0] == speech_lengths.shape[0] == 1

        # encoder_out: (1, maxlen, encoder_dim), len(hyps) = beam_size
        hyps, encoder_out = self._ctc_prefix_beam_search(
            speech, speech_lengths, beam_size, device, params)

        assert len(hyps) == beam_size
        hyps_pad = pad_sequence([
            torch.tensor(hyp[0], device=device, dtype=torch.long)
            for hyp in hyps
        ], True, self.ignore_id)  # (beam_size, max_hyps_len)
        ori_hyps_pad = hyps_pad
        hyps_lens = torch.tensor([len(hyp[0]) for hyp in hyps],
                                 device=device,
                                 dtype=torch.long)  # (beam_size,)
        hyps_pad, _ = add_sos_eos(hyps_pad, self.sos, self.eos, self.ignore_id)
        hyps_lens = hyps_lens + 1  # Add <sos> at begining
        encoder_out = encoder_out.repeat(beam_size, 1, 1)
        encoder_mask = torch.ones(beam_size,
                                  1,
                                  encoder_out.size(1),
                                  dtype=torch.bool,
                                  device=device)

        decoder_out, _ = self.decoder(encoder_out, encoder_mask, hyps_pad, hyps_lens)  
        # (beam_size, max_hyps_len, vocab_size)
        decoder_out = torch.nn.functional.log_softmax(decoder_out, dim=-1)
        decoder_out = decoder_out.cpu().numpy()

        # Only use decoder score for rescoring
        best_score = -float('inf')
        best_index = 0
        for i, hyp in enumerate(hyps):
            score = 0.0
            for j, w in enumerate(hyp[0]):
                score += decoder_out[i][j][w]
            score += decoder_out[i][len(hyp[0])][self.eos]

            # add ctc score
            score += hyp[1] * ctc_weight
            if score > best_score:
                best_score = score
                best_index = i
        return hyps[best_index][0], best_score





def init_stream_asr_model(configs):
    input_dim = configs['input_dim']
    vocab_size = configs['output_dim']
    d_model = configs['encoder_conf']['d_model']
    decoder_type = configs.get('decoder', 'bitransformer')
    
    encoder = Emformer(num_features=input_dim, **configs['encoder_conf'])
    if decoder_type == 'transformer':
        decoder = TransformerDecoder(vocab_size, d_model, **configs['decoder_conf'])
    else:
        assert 0.0 < configs['model_conf']['reverse_weight'] < 1.0
        assert configs['decoder_conf']['r_num_blocks'] > 0
        decoder = BiTransformerDecoder(vocab_size, d_model, **configs['decoder_conf'])
    ctc = CTC(vocab_size, encoder_output_size=d_model)
    
    predictor = Predictor(odim=vocab_size, **configs['decoder_lstm_conf'])
    joint_network = JointNetwork(vocab_size, d_model, d_model, d_model)

    model = StreamASRModel(
        vocab_size=vocab_size,
        encoder=encoder,
        decoder=decoder,
        ctc=ctc,
        predictor=predictor,
        joint_network=joint_network,
        **configs['model_conf']
    )
    model = initializer(model)
    return model
