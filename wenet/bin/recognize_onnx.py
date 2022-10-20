#!/usr/bin/env python3

from __future__ import print_function

import argparse
import os
import copy
import sys

import torch
import yaml
import numpy as np
import torchaudio

from wenet.emformer.asr_model import init_stream_asr_model
from wenet.utils.checkpoint import load_checkpoint
from wenet.bin.stream import Stream, stack_states, unstack_states
from wenet.emformer.scaling_converter import convert_scaled_to_non_scaled
from typing import List, Optional, Tuple
from wenet.emformer.ctc import CTC
from wenet.emformer.emformer import Emformer
import torch.nn.functional as F
from wenet.utils.common import IGNORE_ID, LOG_EPS, add_sos_eos, log_add

import onnx
import onnxruntime
from onnxruntime.quantization import quantize_dynamic, QuantType

def get_args():
    parser = argparse.ArgumentParser(description='export your script model')
    parser.add_argument('--config', required=True, help='config file')
    parser.add_argument('--checkpoint', required=True, help='checkpoint model')
    parser.add_argument('--output_dir', required=True, help='output directory')
    args = parser.parse_args()
    return args


def to_numpy(tensor):
    if tensor.requires_grad:
        return tensor.detach().cpu().numpy()
    else:
        return tensor.cpu().numpy()


class EncoderCtc(torch.nn.Module):
    def __init__(self,
                 encoder: Emformer,
                 ctc: CTC):
        super().__init__()
        self.encoder = encoder
        self.ctc = ctc

    def forward(self, 
            x: torch.Tensor,
            x_lens: torch.Tensor,
            num_processed_frames: torch.Tensor,
            memory_caches: torch.Tensor,
            left_key_caches: torch.Tensor,
            left_val_caches: torch.Tensor,
            conv_caches: torch.Tensor,
    ) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, 
        torch.Tensor, torch.Tensor, torch.Tensor,
    ]:
        """
        B: batch size;
        D: feature dimension;
        T: length of utterance.
        Args:
          x (torch.Tensor): (B, T, D).
          x_lens (torch.Tensor): (B,)
          num_processed_frames: (1,)
          memory_caches: (num_encoder_layers, memory_size, d_model)
          left_key_caches: (num_encoder_layers, left_context_length, d_model)
          left_val_caches: (num_encoder_layers, left_context_length, d_model)
          conv_caches: (num_encoder_layers, d_model, cnn_module_kernel - 1)
        Returns:
          ctc_log_probs:
          output_lengths:
          out_memory_caches:
          out_left_key_caches:
          out_left_val_caches:
          output_conv_caches:
        """
        (
            output, 
            output_lengths,
            out_memory_caches, 
            out_left_key_caches, 
            out_left_val_caches, 
            output_conv_caches
        ) = self.encoder.infer(
            x=x,
            x_lens=x_lens,
            num_processed_frames=num_processed_frames,
            memory_caches = memory_caches,
            left_key_caches=left_key_caches,
            left_val_caches=left_val_caches,
            conv_caches=conv_caches
        )
        ctc_log_probs = self.ctc.log_softmax(output)

        return ctc_log_probs, output_lengths, out_memory_caches, \
               out_left_key_caches, out_left_val_caches, output_conv_caches

def main():
    torch.manual_seed(777)
    args = get_args()
    output_dir = args.output_dir
    os.system("mkdir -p " + output_dir)
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    device = torch.device('cpu')

    with open(args.config, 'r') as fin:
        configs = yaml.load(fin, Loader=yaml.FullLoader)
    params = copy.deepcopy(configs['encoder_conf'])
    params['encoder_dim'] = configs['encoder_conf']['d_model']
    params['output_dim'] = configs['output_dim']

    encoder_outpath = os.path.join(output_dir, 'emformer.quant.onnx')
    ort_session = onnxruntime.InferenceSession(encoder_outpath)

    audio = '/netdisk1/wangzhou/data/aishell_set//data_aishell/wav/test/S0764/BAC009S0764W0121.wav'
    waveform, sr = torchaudio.load(audio, frame_offset=0 , num_frames=-1, normalize=True, channels_first=True)
    assert sr == 16000
    waveform = waveform * (1 << 15)
    kaldi_feat = torchaudio.compliance.kaldi.fbank(
                waveform,
                num_mel_bins=80,
                frame_length=25,
                frame_shift=10,
                dither=False,
                energy_floor=0.0,
                sample_frequency=16000)


    stream = Stream(params=params)
    stream.set_feature(kaldi_feat)
    tail_length = 3 * params['subsampling_factor'] + params['right_context_length'] + 3
    memory_caches, left_key_caches, left_val_caches, conv_caches = stream.states
    memory_caches = to_numpy(memory_caches)
    left_key_caches = to_numpy(left_key_caches)
    left_val_caches = to_numpy(left_val_caches)
    conv_caches = to_numpy(conv_caches)

    result = []
    while not stream.done:
        num_processed_frames = torch.tensor(stream.num_processed_frames, device=device).unsqueeze(0)
        x = stream.get_feature_chunk().unsqueeze(0) # (1, chunk, 80)
        x_lens = torch.tensor(x.size(1), device=device).unsqueeze(0)
        if x.size(1) < tail_length:
            pad_length = tail_length - x.size(1)
            x_lens += pad_length
            x = F.pad(x, (0, 0, 0, pad_length), mode="constant", value=LOG_EPS)
        x = to_numpy(x)
        x_lens = to_numpy(x_lens)
        num_processed_frames = to_numpy(num_processed_frames)

        ort_inputs = {
            'x': x,
            'x_lens': (x_lens),
            'num_processed_frames': (num_processed_frames),
            'memory_caches': (memory_caches),
            'left_key_caches': (left_key_caches),
            'left_val_caches': (left_val_caches),
            'conv_caches': (conv_caches)
        }

        onnx_output = ort_session.run(None, ort_inputs)
        output, _, memory_caches, left_key_caches, left_val_caches, conv_caches = onnx_output

        topk_prob, topk_index = torch.tensor(output).topk(1, dim=2)  # (B, maxlen, 1)
        topk_index = topk_index[0].view(-1)  # (B, maxlen)
        result += topk_index.tolist()
        end = 1

    print(result)


if __name__ == '__main__':
    main()
