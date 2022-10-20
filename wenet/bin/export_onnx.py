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

    with open(args.config, 'r') as fin:
        configs = yaml.load(fin, Loader=yaml.FullLoader)
    params = copy.deepcopy(configs['encoder_conf'])
    params['encoder_dim'] = configs['encoder_conf']['d_model']
    params['output_dim'] = configs['output_dim']

    model = init_stream_asr_model(configs)
    load_checkpoint(model, args.checkpoint)
    encoder_outpath = os.path.join(output_dir, 'emformer.onnx')

    model.eval()
    encoder = convert_scaled_to_non_scaled(model.encoder, inplace=True)
    encoder_ctc = EncoderCtc(encoder, model.ctc)


    speech = torch.randn(1, 100, 80)
    stream = Stream(params=params)
    stream.set_feature(speech[0])

    num_processed_frames = torch.tensor(stream.num_processed_frames).unsqueeze(0)
    x = stream.get_feature_chunk()
    x = x.unsqueeze(0)
    x_lens = torch.tensor(x.size(1)).unsqueeze(0)
    memory_caches, left_key_caches, left_val_caches, conv_caches = stream.states

    (
        output, 
        output_lengths,
        out_memory_caches, 
        out_left_key_caches, 
        out_left_val_caches, 
        output_conv_caches
    )= encoder_ctc(
        x=x,
        x_lens=x_lens,
        num_processed_frames=num_processed_frames,
        memory_caches = memory_caches,
        left_key_caches=left_key_caches,
        left_val_caches=left_val_caches,
        conv_caches=conv_caches
    )

    torch.onnx.export(
        encoder_ctc,
        (x, x_lens, num_processed_frames,
        memory_caches, left_key_caches, left_val_caches, conv_caches),
        encoder_outpath,
        verbose=False,
        opset_version=13,
        input_names=["x", 
                    "x_lens", 
                    "num_processed_frames",
                    "memory_caches", 
                    "left_key_caches", 
                    "left_val_caches", 
                    "conv_caches"],
        output_names=["output", 
                    "output_lengths", 
                    "out_memory_caches", 
                    "out_left_key_caches",
                    "out_left_val_caches", 
                    "output_conv_caches"],
        dynamic_axes={'x': {1: 'T'}, 'output': {1: 'T_OUT'}},
    )


    ort_session = onnxruntime.InferenceSession(encoder_outpath)
    ort_inputs = {
        'x': to_numpy(x),
        'x_lens': to_numpy(x_lens),
        'num_processed_frames': to_numpy(num_processed_frames),
        'memory_caches': to_numpy(memory_caches),
        'left_key_caches': to_numpy(left_key_caches),
        'left_val_caches': to_numpy(left_val_caches),
        'conv_caches': to_numpy(conv_caches)
    }
    onnx_output = ort_session.run(None, ort_inputs)
    np.testing.assert_allclose(to_numpy(output), onnx_output[0],
                               rtol=1e-03, atol=1e-05)

    model_fp32 = os.path.join(output_dir, 'emformer.onnx')
    model_quant = os.path.join(output_dir, 'emformer.quant.onnx')
    quantized_model = quantize_dynamic(model_fp32, model_quant, weight_type=QuantType.QUInt8)

    print('pass, result looks good!')


if __name__ == '__main__':
    main()
