#!/usr/bin/env python3
# Copyright (c) 2022, Xingchen Song (sxc19@mails.tsinghua.edu.cn)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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

try:
    import onnx
    import onnxruntime
except ImportError:
    print('Please install onnxruntime!')
    sys.exit(1)


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


def export_encoder(asr_model, args):
    print("Stage-1: export encoder")
    encoder = asr_model.encoder
    encoder.forward = encoder.forward_chunk
    encoder_outpath = os.path.join(args['output_dir'], 'encoder.onnx')

    print("\tStage-1.1: prepare inputs for encoder")
    chunk = torch.randn(
        (args['batch'], args['decoding_window'], args['feature_size']))
    offset = 0
    # NOTE(xcsong): The uncertainty of `next_cache_start` only appears
    #   in the first few chunks, this is caused by dynamic att_cache shape, i,e
    #   (0, 0, 0, 0) for 1st chunk and (elayers, head, ?, d_k*2) for subsequent
    #   chunks. One way to ease the ONNX export is to keep `next_cache_start`
    #   as a fixed value. To do this, for the **first** chunk, if
    #   left_chunks > 0, we feed real cache & real mask to the model, otherwise
    #   fake cache & fake mask. In this way, we get:
    #   1. 16/-1 mode: next_cache_start == 0 for all chunks
    #   2. 16/4  mode: next_cache_start == chunk_size for all chunks
    #   3. 16/0  mode: next_cache_start == chunk_size for all chunks
    #   4. -1/-1 mode: next_cache_start == 0 for all chunks
    #   NO MORE DYNAMIC CHANGES!!
    if args['left_chunks'] > 0:  # 16/4
        required_cache_size = args['chunk_size'] * args['left_chunks']
        offset = required_cache_size
        # Real cache
        att_cache = torch.zeros(
            (args['num_blocks'], args['head'], required_cache_size,
             args['output_size'] // args['head'] * 2))
        # Real mask
        att_mask = torch.ones(
            (args['batch'], 1, required_cache_size + args['chunk_size']),
            dtype=torch.bool)
        att_mask[:, :, :required_cache_size] = 0
    elif args['left_chunks'] <= 0:  # 16/-1, -1/-1, 16/0
        required_cache_size = -1 if args['left_chunks'] < 0 else 0
        # Fake cache
        att_cache = torch.zeros(
            (args['num_blocks'], args['head'], 0,
             args['output_size'] // args['head'] * 2))
        # Fake mask
        att_mask = torch.ones((0, 0, 0), dtype=torch.bool)
    cnn_cache = torch.zeros(
        (args['num_blocks'], args['batch'],
         args['output_size'], args['cnn_module_kernel'] - 1))
    inputs = (chunk, offset, required_cache_size,
              att_cache, cnn_cache, att_mask)
    print("\t\tchunk.size(): {}\n".format(chunk.size()),
          "\t\toffset: {}\n".format(offset),
          "\t\trequired_cache: {}\n".format(required_cache_size),
          "\t\tatt_cache.size(): {}\n".format(att_cache.size()),
          "\t\tcnn_cache.size(): {}\n".format(cnn_cache.size()),
          "\t\tatt_mask.size(): {}\n".format(att_mask.size()))

    print("\tStage-1.2: torch.onnx.export")
    dynamic_axes = {
        'chunk': {1: 'T'},
        'att_cache': {2: 'T_CACHE'},
        'output': {1: 'T'},
        'r_att_cache': {2: 'T_CACHE'},
    }
    if args['chunk_size'] > 0:  # 16/4, 16/-1, 16/0
        dynamic_axes.pop('chunk')
        dynamic_axes.pop('output')
    if args['left_chunks'] >= 0:  # 16/4, 16/0
        # NOTE(xsong): since we feed real cache & real mask into the
        #   model when left_chunks > 0, the shape of cache will never
        #   be changed.
        dynamic_axes.pop('att_cache')
        dynamic_axes.pop('r_att_cache')
    torch.onnx.export(
        encoder, inputs, encoder_outpath, opset_version=13,
        export_params=True, do_constant_folding=True,
        input_names=[
            'chunk', 'offset', 'required_cache_size',
            'att_cache', 'cnn_cache', 'att_mask'
        ],
        output_names=['output', 'r_att_cache', 'r_cnn_cache'],
        dynamic_axes=dynamic_axes, verbose=False)
    onnx_encoder = onnx.load(encoder_outpath)
    for (k, v) in args.items():
        meta = onnx_encoder.metadata_props.add()
        meta.key, meta.value = str(k), str(v)
    onnx.checker.check_model(onnx_encoder)
    onnx.helper.printable_graph(onnx_encoder.graph)
    # NOTE(xcsong): to add those metadatas we need to reopen
    #   the file and resave it.
    onnx.save(onnx_encoder, encoder_outpath)
    print("\t\tonnx_encoder inputs : {}".format(
        [node.name for node in onnx_encoder.graph.input]))
    print("\t\tonnx_encoder outputs: {}".format(
        [node.name for node in onnx_encoder.graph.output]))
    print('\t\tExport onnx_encoder, done! see {}'.format(
        encoder_outpath))

    print("\tStage-1.3: check onnx_encoder and torch_encoder")
    torch_output = []
    torch_chunk = copy.deepcopy(chunk)
    torch_offset = copy.deepcopy(offset)
    torch_required_cache_size = copy.deepcopy(required_cache_size)
    torch_att_cache = copy.deepcopy(att_cache)
    torch_cnn_cache = copy.deepcopy(cnn_cache)
    torch_att_mask = copy.deepcopy(att_mask)
    for i in range(10):
        print("\t\ttorch chunk-{}: {}, offset: {}, att_cache: {},"
              " cnn_cache: {}, att_mask: {}".format(
                  i, list(torch_chunk.size()), torch_offset,
                  list(torch_att_cache.size()),
                  list(torch_cnn_cache.size()), list(torch_att_mask.size())))
        # NOTE(xsong): att_mask of the first few batches need changes if
        #   we use 16/4 mode.
        if args['left_chunks'] > 0:  # 16/4
            torch_att_mask[:, :, -(args['chunk_size'] * (i + 1)):] = 1
        out, torch_att_cache, torch_cnn_cache = encoder(
            torch_chunk, torch_offset, torch_required_cache_size,
            torch_att_cache, torch_cnn_cache, torch_att_mask)
        torch_output.append(out)
        torch_offset += out.size(1)
    torch_output = torch.cat(torch_output, dim=1)

    onnx_output = []
    onnx_chunk = to_numpy(chunk)
    onnx_offset = np.array((offset)).astype(np.int64)
    onnx_required_cache_size = np.array((required_cache_size)).astype(np.int64)
    onnx_att_cache = to_numpy(att_cache)
    onnx_cnn_cache = to_numpy(cnn_cache)
    onnx_att_mask = to_numpy(att_mask)
    ort_session = onnxruntime.InferenceSession(encoder_outpath)
    for i in range(10):
        print("\t\tonnx  chunk-{}: {}, offset: {}, att_cache: {},"
              " cnn_cache: {}, att_mask: {}".format(
                  i, onnx_chunk.shape, onnx_offset, onnx_att_cache.shape,
                  onnx_cnn_cache.shape, onnx_att_mask.shape))
        # NOTE(xsong): att_mask of the first few batches need changes if
        #   we use 16/4 mode.
        if args['left_chunks'] > 0:  # 16/4
            onnx_att_mask[:, :, -(args['chunk_size'] * (i + 1)):] = 1
        ort_inputs = {
            'chunk': onnx_chunk, 'offset': onnx_offset,
            'required_cache_size': onnx_required_cache_size,
            'att_cache': onnx_att_cache, 'cnn_cache': onnx_cnn_cache,
            'att_mask': onnx_att_mask
        }
        # NOTE(xcsong): If we use 16/-1, -1/-1 or 16/0 mode, `next_cache_start`
        #   will be hardcoded to 0 or chunk_size by ONNX, thus
        #   required_cache_size and att_mask are no more needed and they will
        #   be removed by ONNX automatically.
        if args['left_chunks'] <= 0:  # 16/-1, -1/-1, 16/0
            ort_inputs.pop('required_cache_size')
            ort_inputs.pop('att_mask')
        if 'conformer' not in args['encoder']:
            ort_inputs.pop('cnn_cache')  # Transformer
        ort_outs = ort_session.run(None, ort_inputs)
        onnx_att_cache, onnx_cnn_cache = ort_outs[1], ort_outs[2]
        onnx_output.append(ort_outs[0])
        onnx_offset += ort_outs[0].shape[1]
    onnx_output = np.concatenate(onnx_output, axis=1)

    np.testing.assert_allclose(to_numpy(torch_output), onnx_output,
                               rtol=1e-03, atol=1e-05)
    meta = ort_session.get_modelmeta()
    print("\t\tcustom_metadata_map={}".format(meta.custom_metadata_map))
    print("\t\tCheck onnx_encoder, pass!")


def export_ctc(asr_model, encoder_out, output_dir):
    print("Stage-2: export ctc")
    ctc = asr_model.ctc
    ctc.forward = ctc.log_softmax
    ctc_outpath = os.path.join(output_dir, 'ctc.onnx')

    print("\tStage-2.1: prepare inputs for ctc")
    hidden = encoder_out

    torch.onnx.export(
        ctc, hidden, ctc_outpath, opset_version=13,
        export_params=True, do_constant_folding=True,
        input_names=['hidden'], 
        output_names=['probs'],
        dynamic_axes={'hidden': {1: 'T'}, 'probs': {1: 'T'}}, 
        verbose=False
    )
    onnx_ctc = onnx.load(ctc_outpath)
    onnx.checker.check_model(onnx_ctc)
    onnx.helper.printable_graph(onnx_ctc.graph)
    # NOTE(xcsong): to add those metadatas we need to reopen
    #   the file and resave it.
    onnx.save(onnx_ctc, ctc_outpath)
    print("\t\tonnx_ctc inputs : {}".format(
        [node.name for node in onnx_ctc.graph.input]))
    print("\t\tonnx_ctc outputs: {}".format(
        [node.name for node in onnx_ctc.graph.output]))
    print('\t\tExport onnx_ctc, done! see {}'.format(
        ctc_outpath))

    print("\tStage-2.3: check onnx_ctc and torch_ctc")
    torch_output = ctc(hidden)
    ort_session = onnxruntime.InferenceSession(ctc_outpath)
    onnx_output = ort_session.run(None, {'hidden' : to_numpy(hidden)})

    np.testing.assert_allclose(to_numpy(torch_output), onnx_output[0],
                               rtol=1e-03, atol=1e-05)
    print("\t\tCheck onnx_ctc, pass!")


class EmformerEncoderLayer(torch.nn.Module):
    def __init__(self, d_model:int):
        super().__init__()

    @torch.jit.export
    def infer(self, attn_cache: List[torch.Tensor]) -> List[torch.Tensor]:
        return attn_cache

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

    # model = init_stream_asr_model(configs)
    # load_checkpoint(model, args.checkpoint)
    encoder_outpath = os.path.join(output_dir, 'emformer.onnx')

    from wenet.emformer.emformer_ import Emformer
    encoder = Emformer(
        num_features=80,
        chunk_length=32,
        subsampling_factor=4,
        d_model=256,
        nhead=4,
        dim_feedforward=2048,
        num_encoder_layers=12,
        cnn_module_kernel=31,
        left_context_length=32,
        right_context_length=8,
        memory_size=32,
    )
    encoder.eval()
    # encoder = torch.jit.script(encoder)
    encoder_model = encoder
    encoder_model.forward = encoder_model.infer
    convert_scaled_to_non_scaled(encoder_model, inplace=True)


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
    )= encoder_model.infer(
        x=x,
        x_lens=x_lens,
        num_processed_frames=num_processed_frames,
        memory_caches = memory_caches,
        left_key_caches=left_key_caches,
        left_val_caches=left_val_caches,
        conv_caches=conv_caches
    )

    torch.onnx.export(
        encoder_model,
        (x, x_lens, num_processed_frames, 
        memory_caches, left_key_caches, left_val_caches, conv_caches),
        encoder_outpath,
        verbose=False,
        opset_version=13,
        input_names=["x", "x_lens", "num_processed_frames",
                    'memory_caches', 'left_key_caches', 'left_val_caches', 'conv_caches'],
        output_names=["output", "output_lengths", 
                'out_memory_caches', 'out_left_key_caches',
                'out_left_val_caches', 'output_conv_caches'],
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
    conv_out = ort_session.run(None, ort_inputs)
    np.testing.assert_allclose(to_numpy(output), conv_out[0],
                               rtol=1e-03, atol=1e-05)

    print('pass')

if __name__ == '__main__':
    main()
