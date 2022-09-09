# Copyright    2022  Xiaomi Corp.        (authors: Fangjun Kuang,
#                                                  Zengwei Yao)
#
# See ../../../../LICENSE for clarification regarding multiple authors
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

import math
from typing import List, Optional, Tuple, Dict

import torch


class Stream(object):
    def __init__(
        self,
        params: Dict,
        device: torch.device = torch.device("cpu"),
        LOG_EPS: float = math.log(1e-10),
    ) -> None:
        """
        Args:
          params:
            It's the return value of :func:`get_params`.
          decoding_graph:
            The decoding graph. Can be either a `k2.trivial_graph` or HLG, Used
            only when --decoding_method is fast_beam_search.
          device:
            The device to run this stream.
        """
        self.device = device
        self.LOG_EPS = LOG_EPS

        # Containing attention caches and convolution caches
        self.states: Optional[
            Tuple[List[List[torch.Tensor]], List[torch.Tensor]]
        ] = None
        # Initailize zero states.
        self.init_states(params)
        self.feature: Optional[torch.Tensor] = None
        # Make sure all feature frames can be used.
        # Add 2 here since we will drop the first and last after subsampling.
        self.chunk_length = params['chunk_length']
        self.pad_length = (
            params['right_context_length'] + 2 * params['subsampling_factor'] + 3
        )
        self.num_frames = 0
        self.num_processed_frames = 0

        # After all feature frames are processed, we set this flag to True
        self._done = False

    def set_feature(self, feature: torch.Tensor) -> None:
        assert feature.dim() == 2, feature.dim()
        self.num_frames = feature.size(0)
        # tail padding
        self.feature = torch.nn.functional.pad(
            feature,
            (0, 0, 0, self.pad_length),
            mode="constant",
            value=self.LOG_EPS,
        )


    def init_states(self, params: Dict) -> None:
        attn_caches = [
            [
                torch.zeros(
                    params['memory_size'], params['encoder_dim'], device=self.device
                ),
                torch.zeros(
                    params['left_context_length'] // params['subsampling_factor'],
                    params['encoder_dim'],
                    device=self.device,
                ),
                torch.zeros(
                    params['left_context_length'] // params['subsampling_factor'],
                    params['encoder_dim'],
                    device=self.device,
                ),
            ]
            for _ in range(params['num_encoder_layers'])
        ]
        conv_caches = [
            torch.zeros(
                params['encoder_dim'],
                params['cnn_module_kernel'] - 1,
                device=self.device,
            )
            for _ in range(params['num_encoder_layers'])
        ]
        self.states = (attn_caches, conv_caches)

    def get_feature_chunk(self) -> torch.Tensor:
        """Get a chunk of feature frames.

        Returns:
          A tensor of shape (ret_length, feature_dim).
        """
        update_length = min(
            self.num_frames - self.num_processed_frames, self.chunk_length
        )
        ret_length = update_length + self.pad_length

        ret_feature = self.feature[
            self.num_processed_frames : self.num_processed_frames + ret_length
        ]
        # Cut off used frames.
        # self.feature = self.feature[update_length:]

        self.num_processed_frames += update_length
        if self.num_processed_frames >= self.num_frames:
            self._done = True

        return ret_feature

    @property
    def done(self) -> bool:
        """Return True if all feature frames are processed."""
        return self._done



def unstack_states(
    states: Tuple[List[List[torch.Tensor]], List[torch.Tensor]]
) -> List[Tuple[List[List[torch.Tensor]], List[torch.Tensor]]]:
    """Unstack the emformer state corresponding to a batch of utterances
    into a list of states, where the i-th entry is the state from the i-th
    utterance in the batch.

    Args:
      states:
        A tuple of 2 elements.
        ``states[0]`` is the attention caches of a batch of utterance.
        ``states[1]`` is the convolution caches of a batch of utterance.
        ``len(states[0])`` and ``len(states[1])`` both eqaul to number of layers.  # noqa

    Returns:
      A list of states.
      ``states[i]`` is a tuple of 2 elements of i-th utterance.
      ``states[i][0]`` is the attention caches of i-th utterance.
      ``states[i][1]`` is the convolution caches of i-th utterance.
      ``len(states[i][0])`` and ``len(states[i][1])`` both eqaul to number of layers.  # noqa
    """

    attn_caches, conv_caches = states
    batch_size = conv_caches[0].size(0)
    num_layers = len(attn_caches)

    list_attn_caches = [None] * batch_size
    for i in range(batch_size):
        list_attn_caches[i] = [[] for _ in range(num_layers)]
    for li, layer in enumerate(attn_caches):
        for s in layer:
            s_list = s.unbind(dim=1)
            for bi, b in enumerate(list_attn_caches):
                b[li].append(s_list[bi])

    list_conv_caches = [None] * batch_size
    for i in range(batch_size):
        list_conv_caches[i] = [None] * num_layers
    for li, layer in enumerate(conv_caches):
        c_list = layer.unbind(dim=0)
        for bi, b in enumerate(list_conv_caches):
            b[li] = c_list[bi]

    ans = [None] * batch_size
    for i in range(batch_size):
        ans[i] = [list_attn_caches[i], list_conv_caches[i]]

    return ans


def stack_states(
    state_list: List[Tuple[List[List[torch.Tensor]], List[torch.Tensor]]]
) -> Tuple[List[List[torch.Tensor]], List[torch.Tensor]]:
    """Stack list of emformer states that correspond to separate utterances
    into a single emformer state so that it can be used as an input for
    emformer when those utterances are formed into a batch.

    Note:
      It is the inverse of :func:`unstack_states`.

    Args:
      state_list:
        Each element in state_list corresponding to the internal state
        of the emformer model for a single utterance.
        ``states[i]`` is a tuple of 2 elements of i-th utterance.
        ``states[i][0]`` is the attention caches of i-th utterance.
        ``states[i][1]`` is the convolution caches of i-th utterance.
        ``len(states[i][0])`` and ``len(states[i][1])`` both eqaul to number of layers.  # noqa

    Returns:
      A new state corresponding to a batch of utterances.
      See the input argument of :func:`unstack_states` for the meaning
      of the returned tensor.
    """
    batch_size = len(state_list)

    attn_caches = []
    for layer in state_list[0][0]:
        if batch_size > 1:
            # Note: We will stack attn_caches[layer][s][] later to get attn_caches[layer][s]  # noqa
            attn_caches.append([[s] for s in layer])
        else:
            attn_caches.append([s.unsqueeze(1) for s in layer])
    for b, states in enumerate(state_list[1:], 1):
        for li, layer in enumerate(states[0]):
            for si, s in enumerate(layer):
                attn_caches[li][si].append(s)
                if b == batch_size - 1:
                    attn_caches[li][si] = torch.stack(
                        attn_caches[li][si], dim=1
                    )

    conv_caches = []
    for layer in state_list[0][1]:
        if batch_size > 1:
            # Note: We will stack conv_caches[layer][] later to get conv_caches[layer]  # noqa
            conv_caches.append([layer])
        else:
            conv_caches.append(layer.unsqueeze(0))
    for b, states in enumerate(state_list[1:], 1):
        for li, layer in enumerate(states[1]):
            conv_caches[li].append(layer)
            if b == batch_size - 1:
                conv_caches[li] = torch.stack(conv_caches[li], dim=0)

    return [attn_caches, conv_caches]

