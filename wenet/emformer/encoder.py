from typing import Tuple, List
import math
import torch
import torch.nn as nn
from wenet.emformer.subsampling import Conv2dSubsampling
from wenet.emformer.encoder_layer import EmformerEncoderLayer
from wenet.utils.mask import make_pad_mask


class EncoderInterface(nn.Module):
    def forward(
        self, x: torch.Tensor, x_lens: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
          x:
            A tensor of shape (batch_size, input_seq_len, num_features)
            containing the input features.
          x_lens:
            A tensor of shape (batch_size,) containing the number of frames
            in `x` before padding.
        Returns:
          Return a tuple containing two tensors:
            - encoder_out, a tensor of (batch_size, out_seq_len, output_dim)
              containing unnormalized probabilities, i.e., the output of a
              linear layer.
            - encoder_out_lens, a tensor of shape (batch_size,) containing
              the number of frames in `encoder_out` before padding.
        """
        raise NotImplementedError("Please implement it in a subclass")


def _gen_attention_mask_block(
    col_widths: List[int],
    col_mask: List[bool],
    num_rows: int,
    device: torch.device,
) -> torch.Tensor:
    assert len(col_widths) == len(
        col_mask
    ), "Length of col_widths must match that of col_mask"

    mask_block = [
        torch.ones(num_rows, col_width, device=device)
        if is_ones_col
        else torch.zeros(num_rows, col_width, device=device)
        for col_width, is_ones_col in zip(col_widths, col_mask)
    ]
    return torch.cat(mask_block, dim=1)


class EmformerEncoder(nn.Module):
    """Implements the Emformer architecture introduced in
    *Emformer: Efficient Memory Transformer Based Acoustic Model for Low Latency
    Streaming Speech Recognition*
    [:footcite:`shi2021emformer`].

    Args:
      d_model (int):
        Input dimension.
      nhead (int):
        Number of attention heads in each emformer layer.
      dim_feedforward (int):
        Hidden layer dimension of each emformer layer's feedforward network.
      num_encoder_layers (int):
        Number of emformer layers to instantiate.
      chunk_length (int):
        Length of each input segment.
      dropout (float, optional):
        Dropout probability. (default: 0.0)
      layer_dropout (float, optional):
        Layer dropout probability. (default: 0.0)
      cnn_module_kernel (int):
        Kernel size of convolution module.
      left_context_length (int, optional):
        Length of left context. (default: 0)
      right_context_length (int, optional):
        Length of right context. (default: 0)
      memory_size (int, optional):
        Number of memory elements to use. (default: 0)
      tanh_on_mem (bool, optional):
        If ``true``, applies tanh to memory elements. (default: ``false``)
      negative_inf (float, optional):
        Value to use for negative infinity in attention weights. (default: -1e8)
    """

    def __init__(
        self,
        chunk_length: int,
        d_model: int = 256,
        nhead: int = 4,
        dim_feedforward: int = 2048,
        num_encoder_layers: int = 12,
        dropout: float = 0.1,
        layer_dropout: float = 0.075,
        cnn_module_kernel: int = 31,
        left_context_length: int = 0,
        right_context_length: int = 0,
        memory_size: int = 0,
        tanh_on_mem: bool = False,
        negative_inf: float = -1e8,
    ):
        super().__init__()

        self.use_memory = memory_size > 0
        self.init_memory_op = nn.AvgPool1d(
            kernel_size=chunk_length,
            stride=chunk_length,
            ceil_mode=True,
        )

        self.emformer_layers = nn.ModuleList(
            [
                EmformerEncoderLayer(
                    d_model=d_model,
                    nhead=nhead,
                    dim_feedforward=dim_feedforward,
                    chunk_length=chunk_length,
                    dropout=dropout,
                    layer_dropout=layer_dropout,
                    cnn_module_kernel=cnn_module_kernel,
                    left_context_length=left_context_length,
                    right_context_length=right_context_length,
                    memory_size=memory_size,
                    tanh_on_mem=tanh_on_mem,
                    negative_inf=negative_inf,
                )
                for layer_idx in range(num_encoder_layers)
            ]
        )

        self.num_encoder_layers = num_encoder_layers
        self.d_model = d_model
        self.left_context_length = left_context_length
        self.right_context_length = right_context_length
        self.chunk_length = chunk_length
        self.memory_size = memory_size
        self.cnn_module_kernel = cnn_module_kernel

    def _gen_right_context(self, x: torch.Tensor) -> torch.Tensor:
        """Hard copy each chunk's right context and concat them."""
        T = x.shape[0]
        num_chunks = math.ceil(
            (T - self.right_context_length) / self.chunk_length
        )
        # first (num_chunks - 1) right context block
        intervals = torch.arange(
            0, self.chunk_length * (num_chunks - 1), self.chunk_length
        )
        first = torch.arange(
            self.chunk_length, self.chunk_length + self.right_context_length
        )
        indexes = intervals.unsqueeze(1) + first.unsqueeze(0)
        # cat last right context block
        indexes = torch.cat(
            [
                indexes,
                torch.arange(T - self.right_context_length, T).unsqueeze(0),
            ]
        )
        right_context_blocks = x[indexes.reshape(-1)]
        return right_context_blocks

    def _gen_attention_mask_col_widths(
        self, chunk_idx: int, U: int
    ) -> List[int]:
        """Calculate column widths (key, value) in attention mask for the
        chunk_idx chunk."""
        num_chunks = math.ceil(U / self.chunk_length)
        rc = self.right_context_length
        lc = self.left_context_length
        rc_start = chunk_idx * rc
        rc_end = rc_start + rc
        chunk_start = max(chunk_idx * self.chunk_length - lc, 0)
        chunk_end = min((chunk_idx + 1) * self.chunk_length, U)
        R = rc * num_chunks

        if self.use_memory:
            m_start = max(chunk_idx - self.memory_size, 0)
            M = num_chunks - 1
            col_widths = [
                m_start,  # before memory
                chunk_idx - m_start,  # memory
                M - chunk_idx,  # after memory
                rc_start,  # before right context
                rc,  # right context
                R - rc_end,  # after right context
                chunk_start,  # before chunk
                chunk_end - chunk_start,  # chunk
                U - chunk_end,  # after chunk
            ]
        else:
            col_widths = [
                rc_start,  # before right context
                rc,  # right context
                R - rc_end,  # after right context
                chunk_start,  # before chunk
                chunk_end - chunk_start,  # chunk
                U - chunk_end,  # after chunk
            ]

        return col_widths

    def _gen_attention_mask(self, utterance: torch.Tensor) -> torch.Tensor:
        """Generate attention mask to simulate underlying chunk-wise attention
        computation, where chunk-wise connections are filled with `False`,
        and other unnecessary connections beyond chunk are filled with `True`.

        R: length of hard-copied right contexts;
        U: length of full utterance;
        S: length of summary vectors;
        M: length of memory vectors;
        Q: length of attention query;
        KV: length of attention key and value.

        The shape of attention mask is (Q, KV).
        If self.use_memory is `True`:
          query = [right_context, utterance, summary];
          key, value = [memory, right_context, utterance];
          Q = R + U + S, KV = M + R + U.
        Otherwise:
          query = [right_context, utterance]
          key, value = [right_context, utterance]
          Q = R + U, KV = R + U.

        Suppose:
          c_i: chunk at index i;
          r_i: right context that c_i can use;
          l_i: left context that c_i can use;
          m_i: past memory vectors from previous layer that c_i can use;
          s_i: summary vector of c_i.
        The target chunk-wise attention is:
          c_i, r_i (in query) -> l_i, c_i, r_i, m_i (in key);
          s_i (in query) -> l_i, c_i, r_i (in key).
        """
        U = utterance.size(0)
        num_chunks = math.ceil(U / self.chunk_length)

        right_context_mask = []
        utterance_mask = []
        summary_mask = []

        if self.use_memory:
            num_cols = 9
            # right context and utterance both attend to memory, right context,
            # utterance
            right_context_utterance_cols_mask = [
                idx in [1, 4, 7] for idx in range(num_cols)
            ]
            # summary attends to right context, utterance
            summary_cols_mask = [idx in [4, 7] for idx in range(num_cols)]
            masks_to_concat = [right_context_mask, utterance_mask, summary_mask]
        else:
            num_cols = 6
            # right context and utterance both attend to right context and
            # utterance
            right_context_utterance_cols_mask = [
                idx in [1, 4] for idx in range(num_cols)
            ]
            summary_cols_mask = None
            masks_to_concat = [right_context_mask, utterance_mask]

        for chunk_idx in range(num_chunks):
            col_widths = self._gen_attention_mask_col_widths(chunk_idx, U)

            right_context_mask_block = _gen_attention_mask_block(
                col_widths,
                right_context_utterance_cols_mask,
                self.right_context_length,
                utterance.device,
            )
            right_context_mask.append(right_context_mask_block)

            utterance_mask_block = _gen_attention_mask_block(
                col_widths,
                right_context_utterance_cols_mask,
                min(
                    self.chunk_length,
                    U - chunk_idx * self.chunk_length,
                ),
                utterance.device,
            )
            utterance_mask.append(utterance_mask_block)

            if summary_cols_mask is not None:
                summary_mask_block = _gen_attention_mask_block(
                    col_widths, summary_cols_mask, 1, utterance.device
                )
                summary_mask.append(summary_mask_block)

        attention_mask = (
            1 - torch.cat([torch.cat(mask) for mask in masks_to_concat])
        ).to(torch.bool)
        return attention_mask

    def forward(
        self, x: torch.Tensor, lengths: torch.Tensor, warmup: float = 1.0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass for training and validation mode.

        B: batch size;
        D: input dimension;
        U: length of utterance.

        Args:
          x (torch.Tensor):
            Utterance frames right-padded with right context frames,
            with shape (U + right_context_length, B, D).
          lengths (torch.Tensor):
            With shape (B,) and i-th element representing number of valid
            utterance frames for i-th batch element in x, which contains the
            right_context at the end.

        Returns:
          A tuple of 2 tensors:
            - output utterance frames, with shape (U, B, D).
            - output_lengths, with shape (B,), without containing the
              right_context at the end.
        """
        U = x.size(0) - self.right_context_length

        right_context = self._gen_right_context(x)
        utterance = x[:U]
        output_lengths = torch.clamp(lengths - self.right_context_length, min=0)
        attention_mask = self._gen_attention_mask(utterance)
        memory = (
            self.init_memory_op(utterance.permute(1, 2, 0)).permute(2, 0, 1)[
                :-1
            ]
            if self.use_memory
            else torch.empty(0).to(dtype=x.dtype, device=x.device)
        )
        padding_mask = make_pad_mask(
            memory.size(0) + right_context.size(0) + output_lengths
        )

        output = utterance
        for layer in self.emformer_layers:
            output, right_context, memory = layer(
                output,
                right_context,
                memory,
                attention_mask,
                padding_mask=padding_mask,
                warmup=warmup,
            )

        return output, output_lengths

    @torch.jit.export
    def infer(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor,
        num_processed_frames: torch.Tensor,
        states: Tuple[List[List[torch.Tensor]], List[torch.Tensor]],
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        Tuple[List[List[torch.Tensor]], List[torch.Tensor]],
    ]:
        """Forward pass for streaming inference.

        B: batch size;
        D: input dimension;
        U: length of utterance.

        Args:
          x (torch.Tensor):
            Utterance frames right-padded with right context frames,
            with shape (U + right_context_length, B, D).
          lengths (torch.Tensor):
            With shape (B,) and i-th element representing number of valid
            utterance frames for i-th batch element in x, which contains the
            right_context at the end.
          states (List[torch.Tensor, List[List[torch.Tensor]], List[torch.Tensor]]: # noqa
            Cached states containing:
            - past_lens: number of past frames for each sample in batch
            - attn_caches: attention states from preceding chunk's computation,
              where each element corresponds to each emformer layer
            - conv_caches: left context for causal convolution, where each
              element corresponds to each layer.

        Returns:
          (Tensor, Tensor, List[List[torch.Tensor]], List[torch.Tensor]):
            - output utterance frames, with shape (U, B, D).
            - output lengths, with shape (B,), without containing the
              right_context at the end.
            - updated states from current chunk's computation.
        """
        assert num_processed_frames.shape == (x.size(1),)

        attn_caches = states[0]
        assert len(attn_caches) == self.num_encoder_layers, len(attn_caches)
        for i in range(len(attn_caches)):
            assert attn_caches[i][0].shape == (
                self.memory_size,
                x.size(1),
                self.d_model,
            ), attn_caches[i][0].shape
            assert attn_caches[i][1].shape == (
                self.left_context_length,
                x.size(1),
                self.d_model,
            ), attn_caches[i][1].shape
            assert attn_caches[i][2].shape == (
                self.left_context_length,
                x.size(1),
                self.d_model,
            ), attn_caches[i][2].shape

        conv_caches = states[1]
        assert len(conv_caches) == self.num_encoder_layers, len(conv_caches)
        for i in range(len(conv_caches)):
            assert conv_caches[i].shape == (
                x.size(1),
                self.d_model,
                self.cnn_module_kernel - 1,
            ), conv_caches[i].shape

        right_context = x[-self.right_context_length :]
        utterance = x[: -self.right_context_length]
        output_lengths = torch.clamp(lengths - self.right_context_length, min=0)
        memory = (
            self.init_memory_op(utterance.permute(1, 2, 0)).permute(2, 0, 1)
            if self.use_memory
            else torch.empty(0).to(dtype=x.dtype, device=x.device)
        )

        # calcualte padding mask to mask out initial zero caches
        chunk_mask = make_pad_mask(output_lengths).to(x.device)
        memory_mask = (
            torch.div(
                num_processed_frames, self.chunk_length, rounding_mode="floor"
            ).view(x.size(1), 1)
            <= torch.arange(self.memory_size, device=x.device).expand(
                x.size(1), self.memory_size
            )
        ).flip(1)
        left_context_mask = (
            num_processed_frames.view(x.size(1), 1)
            <= torch.arange(self.left_context_length, device=x.device).expand(
                x.size(1), self.left_context_length
            )
        ).flip(1)
        right_context_mask = torch.zeros(
            x.size(1),
            self.right_context_length,
            dtype=torch.bool,
            device=x.device,
        )
        padding_mask = torch.cat(
            [memory_mask, right_context_mask, left_context_mask, chunk_mask],
            dim=1,
        )

        output = utterance
        output_attn_caches: List[List[torch.Tensor]] = []
        output_conv_caches: List[torch.Tensor] = []
        for layer_idx, layer in enumerate(self.emformer_layers):
            (
                output,
                right_context,
                memory,
                output_attn_cache,
                output_conv_cache,
            ) = layer.infer(
                output,
                right_context,
                memory,
                padding_mask=padding_mask,
                attn_cache=attn_caches[layer_idx],
                conv_cache=conv_caches[layer_idx],
            )
            output_attn_caches.append(output_attn_cache)
            output_conv_caches.append(output_conv_cache)

        output_states: Tuple[List[List[torch.Tensor]], List[torch.Tensor]] = (
            output_attn_caches,
            output_conv_caches,
        )
        return output, output_lengths, output_states


class Emformer(EncoderInterface):
    def __init__(
        self,
        num_features: int,
        chunk_length: int,
        subsampling_factor: int = 4,
        d_model: int = 256,
        nhead: int = 4,
        dim_feedforward: int = 2048,
        num_encoder_layers: int = 12,
        dropout: float = 0.1,
        layer_dropout: float = 0.075,
        cnn_module_kernel: int = 3,
        left_context_length: int = 0,
        right_context_length: int = 0,
        memory_size: int = 0,
        tanh_on_mem: bool = False,
        negative_inf: float = -1e8,
    ):
        super().__init__()

        self.subsampling_factor = subsampling_factor
        self.right_context_length = right_context_length
        if subsampling_factor != 4:
            raise NotImplementedError("Support only 'subsampling_factor=4'.")
        if chunk_length % subsampling_factor != 0:
            raise NotImplementedError(
                "chunk_length must be a mutiple of subsampling_factor."
            )
        if (
            left_context_length != 0
            and left_context_length % subsampling_factor != 0
        ):
            raise NotImplementedError(
                "left_context_length must be 0 or a mutiple of subsampling_factor."  # noqa
            )
        if (
            right_context_length != 0
            and right_context_length % subsampling_factor != 0
        ):
            raise NotImplementedError(
                "right_context_length must be 0 or a mutiple of subsampling_factor."  # noqa
            )

        # self.encoder_embed converts the input of shape (N, T, num_features)
        # to the shape (N, T//subsampling_factor, d_model).
        # That is, it does two things simultaneously:
        #   (1) subsampling: T -> T//subsampling_factor
        #   (2) embedding: num_features -> d_model
        self.encoder_embed = Conv2dSubsampling(num_features, d_model)

        self.encoder = EmformerEncoder(
            chunk_length=chunk_length // subsampling_factor,
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            num_encoder_layers=num_encoder_layers,
            dropout=dropout,
            layer_dropout=layer_dropout,
            cnn_module_kernel=cnn_module_kernel,
            left_context_length=left_context_length // subsampling_factor,
            right_context_length=right_context_length // subsampling_factor,
            memory_size=memory_size,
            tanh_on_mem=tanh_on_mem,
            negative_inf=negative_inf,
        )

    def forward(
        self, x: torch.Tensor, x_lens: torch.Tensor, warmup: float = 1.0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass for training and non-streaming inference.

        B: batch size;
        D: feature dimension;
        T: length of utterance.

        Args:
          x (torch.Tensor):
            Utterance frames right-padded with right context frames,
            with shape (B, T, D).
          x_lens (torch.Tensor):
            With shape (B,) and i-th element representing number of valid
            utterance frames for i-th batch element in x, containing the
            right_context at the end.
          warmup:
            A floating point value that gradually increases from 0 throughout
            training; when it is >= 1.0 we are "fully warmed up".  It is used
            to turn modules on sequentially.

        Returns:
          (Tensor, Tensor):
            - output embedding, with shape (B, T', D), where
              T' = ((T - 1) // 2 - 1) // 2 - self.right_context_length // 4.
            - output lengths, with shape (B,), without containing the
              right_context at the end.
        """
        x = self.encoder_embed(x)

        x = x.permute(1, 0, 2)  # (N, T, C) -> (T, N, C)

        x_lens = (((x_lens - 1) >> 1) - 1) >> 1
        assert x.size(0) == x_lens.max().item()

        output, output_lengths = self.encoder(
            x, x_lens, warmup=warmup
        )  # (T, N, C)

        output = output.permute(1, 0, 2)  # (T, N, C) -> (N, T, C)

        return output, output_lengths

    @torch.jit.export
    def infer(
        self,
        x: torch.Tensor,
        x_lens: torch.Tensor,
        num_processed_frames: torch.Tensor,
        states: Tuple[List[List[torch.Tensor]], List[torch.Tensor]],
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        Tuple[List[List[torch.Tensor]], List[torch.Tensor]],
    ]:
        """Forward pass for streaming inference.

        B: batch size;
        D: feature dimension;
        T: length of utterance.

        Args:
          x (torch.Tensor):
            Utterance frames right-padded with right context frames,
            with shape (B, T, D).
          lengths (torch.Tensor):
            With shape (B,) and i-th element representing number of valid
            utterance frames for i-th batch element in x, containing the
            right_context at the end.
          states (List[torch.Tensor, List[List[torch.Tensor]], List[torch.Tensor]]: # noqa
            Cached states containing:
            - past_lens: number of past frames for each sample in batch
            - attn_caches: attention states from preceding chunk's computation,
              where each element corresponds to each emformer layer
            - conv_caches: left context for causal convolution, where each
              element corresponds to each layer.
        Returns:
          (Tensor, Tensor):
            - output embedding, with shape (B, T', D), where
              T' = ((T - 1) // 2 - 1) // 2 - self.right_context_length // 4.
            - output lengths, with shape (B,), without containing the
              right_context at the end.
            - updated states from current chunk's computation.
        """
        x = self.encoder_embed(x)
        # drop the first and last frames
        x = x[:, 1:-1, :]
        x = x.permute(1, 0, 2)  # (N, T, C) -> (T, N, C)

        # Caution: We assume the subsampling factor is 4!
        x_lens = (((x_lens - 1) >> 1) - 1) >> 1
        x_lens -= 2
        assert x.size(0) == x_lens.max().item()

        num_processed_frames = num_processed_frames >> 2

        output, output_lengths, output_states = self.encoder.infer(
            x, x_lens, num_processed_frames, states
        )

        output = output.permute(1, 0, 2)  # (T, N, C) -> (N, T, C)

        return output, output_lengths, output_states

