#!/usr/bin/env python3

from __future__ import print_function

import argparse
import os
import copy
import json

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
from wenet.utils.common import remove_duplicates_and_blank

import onnx
import onnxruntime
from onnxruntime.quantization import quantize_dynamic, QuantType

def get_args():
    parser = argparse.ArgumentParser(description='export your script model')
    parser.add_argument('--config', required=True, help='config file')
    parser.add_argument('--dict', required=True, help='dict')
    parser.add_argument('--output_dir', required=True, help='output directory')
    parser.add_argument('--test', required=True, help='test')
    parser.add_argument('--result_file', required=True, help='result_file')
    args = parser.parse_args()
    return args


def to_numpy(tensor):
    if tensor.requires_grad:
        return tensor.detach().cpu().numpy()
    else:
        return tensor.cpu().numpy()


def main():
    torch.manual_seed(777)
    args = get_args()
    output_dir = args.output_dir
    os.system("mkdir -p " + output_dir)
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    device = torch.device('cpu')

    words = {}
    with open(args.dict, 'r') as fin:
        for line in fin.readlines():
            word, idx = line.strip().split()
            words[int(idx)] = word

    with open(args.config, 'r') as fin:
        configs = yaml.load(fin, Loader=yaml.FullLoader)
    params = copy.deepcopy(configs['encoder_conf'])
    params['encoder_dim'] = configs['encoder_conf']['d_model']
    params['output_dim'] = configs['output_dim']

    # encoder_outpath = os.path.join(output_dir, 'emformer.onnx')
    encoder_outpath = os.path.join(output_dir, 'emformer.onnx')
    ort_session = onnxruntime.InferenceSession(encoder_outpath)
    fout = open(args.result_file, 'w')

    with open(args.test, 'r') as f:
        for line in f.readlines()[1:2]:
            print(line)
            js = json.loads(line.strip())
            key_ = js["key"]
            audio = js["wav"]
            waveform, sr = torchaudio.load(audio, frame_offset=0 , num_frames=-1, normalize=True, channels_first=True)
            assert sr == 16000
            waveform = waveform * (1 << 15)
            kaldi_feat = torchaudio.compliance.kaldi.fbank(
                        waveform,
                        num_mel_bins=80,
                        frame_length=25,
                        frame_shift=10,
                        dither=False,
                        sample_frequency=16000)

            stream = Stream(params=params)
            stream.set_feature(kaldi_feat)
            tail_length = 3 * params['subsampling_factor'] + params['right_context_length'] + 5
            memory_caches, left_key_caches, left_val_caches, conv_caches = stream.states
            memory_caches = to_numpy(memory_caches)
            left_key_caches = to_numpy(left_key_caches)
            left_val_caches = to_numpy(left_val_caches)
            conv_caches = to_numpy(conv_caches)

            result = []
            num_ = 0
            while not stream.done:
                num_processed_frames = torch.tensor(stream.num_processed_frames, device=device).unsqueeze(0)
                x = stream.get_feature_chunk().unsqueeze(0) # (1, chunk, 80)
                x_lens = torch.tensor(x.size(1), device=device).unsqueeze(0)
                print(num_, x.shape, x_lens)
                for i in range(x.shape[1]):
                    print(i, x[0,i,0])
                num_ += 1
                if x.size(1) < tail_length:
                    pad_length = tail_length - x.size(1)
                    x_lens += pad_length
                    x = F.pad(x, (0, 0, 0, pad_length), mode="constant", value=0)
                x = to_numpy(x)
                x_lens = to_numpy(x_lens)
                num_processed_frames = to_numpy(num_processed_frames)

                print("input:", x[0,0,0], x_lens, num_processed_frames, 
                    memory_caches[0,0,0,0], left_key_caches[0,0,0,0], 
                    left_val_caches[0,0,0,0], conv_caches[0,0,0,0])

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
                ss = torch.tensor(output)[0]
                for i in range(ss.shape[0]):
                    print("output:", ss[i, 0].item(), memory_caches[0,0,0,0], left_key_caches[0,0,0,0],
                        left_val_caches[0,0,0,0], conv_caches[0,0,0,0])

                _, topk_index = torch.tensor(output).topk(1, dim=2)  # (B, maxlen, 1)
                topk_index = topk_index[0].view(-1)  # (B, maxlen)
                result += topk_index.tolist()
            result = remove_duplicates_and_blank(result)
            rec = "".join([words[x] for x in result])
            fout.write("%s %s\n" %(key_, rec))
    fout.close()


if __name__ == '__main__':
    main()
