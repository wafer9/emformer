from collections import defaultdict
from typing import List, Optional, Tuple, Dict

import torch

from wenet.emformer.ctc import CTC
from wenet.emformer.emformer import Emformer

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
        ctc: CTC,
        predictor: Predictor,
        joint_network: JointNetwork,
        ctc_weight: float = 0.5,
        ignore_id: int = IGNORE_ID,
        reverse_weight: float = 0.0,
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
        self.ctc = ctc

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
        encoder_out, encoder_out_lens = self.encoder(speech, speech_lengths, warmup=warmup)
        encoder_mask = ~make_pad_mask(encoder_out_lens, encoder_out.size(1)).unsqueeze(1) 

        # ctc
        loss_ctc = self.ctc(encoder_out, encoder_out_lens, text, text_lengths)

        # transducer
        predict_ys_in_pad, target, target_len = prepare_loss_inputs(text, encoder_mask)
        predictor_out = self.predictor(predict_ys_in_pad)
        h_enc = encoder_out.unsqueeze(2)
        h_dec = predictor_out.unsqueeze(1)
        joint_out = self.joint_network(h_enc, h_dec)
        target = target.to(dtype=torch.int32)
        encoder_out_lens = encoder_out_lens.to(dtype=torch.int32)
        loss_trans = self.transducer_loss(joint_out, target, encoder_out_lens, target_len)

        loss = loss_ctc * self.ctc_weight * loss_trans * (1 - self.ctc_weight)

        return loss, None, loss_ctc, loss_trans


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


    def trans_greedy_search(
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
        context_size = 2

        speech = torch.nn.functional.pad(
            speech,
            pad=(0, 0, 0, params['right_context_length']),
            value=LOG_EPS,
        )
        encoder_out, encoder_out_lens = self.encoder(x=speech, x_lens=speech_lengths)
        
        decoder_input = torch.tensor([self.blank_id] * context_size, device=device, dtype=torch.int64).reshape(1, context_size)
        decoder_out = self.predictor(decoder_input, need_pad=False)

        T = encoder_out.size(1)
        hyp = [self.blank_id] * context_size
        t = 0
        max_sym_per_frame = 5
        # Maximum symbols per utterance.
        max_sym_per_utt = 1000

        # symbols per frame
        sym_per_frame = 0

        # symbols per utterance decoded so far
        sym_per_utt = 0

        while t < T and sym_per_utt < max_sym_per_utt:
            if sym_per_frame >= max_sym_per_frame:
                sym_per_frame = 0
                t += 1
                continue
            current_encoder_out = encoder_out[:, t:t+1, :].unsqueeze(2)
            logits = self.joint_network(current_encoder_out, decoder_out.unsqueeze(1))
            y = logits.argmax().item()
            if y != self.blank_id:
                hyp.append(y)
                decoder_input = torch.tensor([hyp[-context_size:]], device=device).reshape(1, context_size)
                decoder_out = self.predictor(decoder_input, need_pad=False)
                sym_per_utt += 1
                sym_per_frame += 1
            else:
                sym_per_frame = 0
                t += 1

        hyp = hyp[context_size:]
        return [hyp]


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



def init_stream_asr_model(configs):
    input_dim = configs['input_dim']
    vocab_size = configs['output_dim']
    encoder_dim = configs['encoder_conf']['d_model']
    
    encoder = Emformer(num_features=input_dim, **configs['encoder_conf'])
    ctc = CTC(vocab_size, encoder_output_size=encoder_dim)
    
    predictor_dim = configs['predictor_conf']['predictor_dim']
    context_size = configs['predictor_conf']['context_size']
    joint_dim = configs['joint_dim']
    predictor = Predictor(vocab_size, predictor_dim, context_size)
    joint_network = JointNetwork(vocab_size, encoder_dim, predictor_dim, joint_dim)

    model = StreamASRModel(
        vocab_size=vocab_size,
        encoder=encoder,
        ctc=ctc,
        predictor=predictor,
        joint_network=joint_network,
        **configs['model_conf']
    )
    model = initializer(model)
    return model
