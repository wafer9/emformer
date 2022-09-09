from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from wenet.emformer.attention import EmformerAttention
from wenet.emformer.convolution import ConvolutionModule

from wenet.emformer.scaling import (
    ActivationBalancer,
    BasicNorm,
    DoubleSwish,
    ScaledConv1d,
    ScaledConv2d,
    ScaledLinear,
)


class EmformerEncoderLayer(nn.Module):
    """Emformer layer that constitutes Emformer.

    Args:
      d_model (int):
        Input dimension.
      nhead (int):
        Number of attention heads.
      dim_feedforward (int):
        Hidden layer dimension of feedforward network.
      chunk_length (int):
        Length of each input segment.
      dropout (float, optional):
        Dropout probability. (Default: 0.0)
      layer_dropout (float, optional):
        Layer dropout probability. (Default: 0.0)
      cnn_module_kernel (int):
        Kernel size of convolution module.
      left_context_length (int, optional):
        Length of left context. (Default: 0)
      right_context_length (int, optional):
        Length of right context. (Default: 0)
      memory_size (int, optional):
        Number of memory elements to use. (Default: 0)
      tanh_on_mem (bool, optional):
        If ``True``, applies tanh to memory elements. (Default: ``False``)
      negative_inf (float, optional):
        Value to use for negative infinity in attention weights. (Default: -1e8)
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        chunk_length: int,
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

        self.attention = EmformerAttention(
            embed_dim=d_model,
            nhead=nhead,
            dropout=dropout,
            tanh_on_mem=tanh_on_mem,
            negative_inf=negative_inf,
        )
        self.summary_op = nn.AvgPool1d(
            kernel_size=chunk_length, stride=chunk_length, ceil_mode=True
        )

        self.feed_forward_macaron = nn.Sequential(
            ScaledLinear(d_model, dim_feedforward),
            ActivationBalancer(channel_dim=-1),
            DoubleSwish(),
            nn.Dropout(dropout),
            ScaledLinear(dim_feedforward, d_model, initial_scale=0.25),
        )

        self.feed_forward = nn.Sequential(
            ScaledLinear(d_model, dim_feedforward),
            ActivationBalancer(channel_dim=-1),
            DoubleSwish(),
            nn.Dropout(dropout),
            ScaledLinear(dim_feedforward, d_model, initial_scale=0.25),
        )

        self.conv_module = ConvolutionModule(
            chunk_length,
            right_context_length,
            d_model,
            cnn_module_kernel,
        )

        self.norm_final = BasicNorm(d_model)

        # try to ensure the output is close to zero-mean
        # (or at least, zero-median).
        self.balancer = ActivationBalancer(
            channel_dim=-1, min_positive=0.45, max_positive=0.55, max_abs=6.0
        )

        self.dropout = nn.Dropout(dropout)

        self.layer_dropout = layer_dropout
        self.left_context_length = left_context_length
        self.chunk_length = chunk_length
        self.memory_size = memory_size
        self.d_model = d_model
        self.use_memory = memory_size > 0

    def _update_attn_cache(
        self,
        next_key: torch.Tensor,
        next_val: torch.Tensor,
        memory: torch.Tensor,
        attn_cache: List[torch.Tensor],
    ) -> List[torch.Tensor]:
        """Update cached attention state:
        1) output memory of current chunk in the lower layer;
        2) attention key and value in current chunk's computation, which would
        be resued in next chunk's computation.
        """
        new_memory = torch.cat([attn_cache[0], memory])
        new_key = torch.cat([attn_cache[1], next_key])
        new_val = torch.cat([attn_cache[2], next_val])
        attn_cache[0] = new_memory[new_memory.size(0) - self.memory_size :]
        attn_cache[1] = new_key[new_key.size(0) - self.left_context_length :]
        attn_cache[2] = new_val[new_val.size(0) - self.left_context_length :]
        return attn_cache

    def _apply_conv_module_forward(
        self,
        right_context_utterance: torch.Tensor,
        R: int,
    ) -> torch.Tensor:
        """Apply convolution module in training and validation mode."""
        utterance = right_context_utterance[R:]
        right_context = right_context_utterance[:R]
        utterance, right_context = self.conv_module(utterance, right_context)
        right_context_utterance = torch.cat([right_context, utterance])
        return right_context_utterance

    def _apply_conv_module_infer(
        self,
        right_context_utterance: torch.Tensor,
        R: int,
        conv_cache: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply convolution module on utterance in inference mode."""
        utterance = right_context_utterance[R:]
        right_context = right_context_utterance[:R]
        utterance, right_context, conv_cache = self.conv_module.infer(
            utterance, right_context, conv_cache
        )
        right_context_utterance = torch.cat([right_context, utterance])
        return right_context_utterance, conv_cache

    def _apply_attention_module_forward(
        self,
        right_context_utterance: torch.Tensor,
        R: int,
        memory: torch.Tensor,
        attention_mask: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply attention module in training and validation mode."""
        utterance = right_context_utterance[R:]
        right_context = right_context_utterance[:R]

        if self.use_memory:
            summary = self.summary_op(utterance.permute(1, 2, 0)).permute(
                2, 0, 1
            )
        else:
            summary = torch.empty(0).to(
                dtype=utterance.dtype, device=utterance.device
            )
        output_right_context_utterance, output_memory = self.attention(
            utterance=utterance,
            right_context=right_context,
            summary=summary,
            memory=memory,
            attention_mask=attention_mask,
            padding_mask=padding_mask,
        )

        return output_right_context_utterance, output_memory

    def _apply_attention_module_infer(
        self,
        right_context_utterance: torch.Tensor,
        R: int,
        memory: torch.Tensor,
        attn_cache: List[torch.Tensor],
        padding_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        """Apply attention module in inference mode.
        1) Unpack cached states including:
           - memory from previous chunks in the lower layer;
           - attention key and value of left context from preceding
             chunk's compuation;
        2) Apply attention computation;
        3) Update cached attention states including:
           - output memory of current chunk in the lower layer;
           - attention key and value in current chunk's computation, which would
             be resued in next chunk's computation.
        """
        utterance = right_context_utterance[R:]
        right_context = right_context_utterance[:R]

        pre_memory = attn_cache[0]
        left_context_key = attn_cache[1]
        left_context_val = attn_cache[2]

        if self.use_memory:
            summary = self.summary_op(utterance.permute(1, 2, 0)).permute(
                2, 0, 1
            )
            summary = summary[:1]
        else:
            summary = torch.empty(0).to(
                dtype=utterance.dtype, device=utterance.device
            )
        (
            output_right_context_utterance,
            output_memory,
            next_key,
            next_val,
        ) = self.attention.infer(
            utterance=utterance,
            right_context=right_context,
            summary=summary,
            memory=pre_memory,
            left_context_key=left_context_key,
            left_context_val=left_context_val,
            padding_mask=padding_mask,
        )
        attn_cache = self._update_attn_cache(
            next_key, next_val, memory, attn_cache
        )
        return output_right_context_utterance, output_memory, attn_cache

    def forward(
        self,
        utterance: torch.Tensor,
        right_context: torch.Tensor,
        memory: torch.Tensor,
        attention_mask: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        warmup: float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        r"""Forward pass for training and validation mode.

        B: batch size;
        D: embedding dimension;
        R: length of hard-copied right contexts;
        U: length of full utterance;
        M: length of memory vectors.

        Args:
          utterance (torch.Tensor):
            Utterance frames, with shape (U, B, D).
          right_context (torch.Tensor):
            Right context frames, with shape (R, B, D).
          memory (torch.Tensor):
            Memory elements, with shape (M, B, D).
            It is an empty tensor without using memory.
          attention_mask (torch.Tensor):
            Attention mask for underlying attention module,
            with shape (Q, KV), where Q = R + U + S, KV = M + R + U.
          padding_mask (torch.Tensor):
            Padding mask of ker tensor, with shape (B, KV).

        Returns:
          A tuple containing 3 tensors:
            - output utterance, with shape (U, B, D).
            - output right context, with shape (R, B, D).
            - output memory, with shape (M, B, D).
        """
        R = right_context.size(0)
        src = torch.cat([right_context, utterance])
        src_orig = src

        warmup_scale = min(0.1 + warmup, 1.0)
        # alpha = 1.0 means fully use this encoder layer, 0.0 would mean
        # completely bypass it.
        if self.training:
            alpha = (
                warmup_scale
                if torch.rand(()).item() <= (1.0 - self.layer_dropout)
                else 0.1
            )
        else:
            alpha = 1.0

        # macaron style feed forward module
        src = src + self.dropout(self.feed_forward_macaron(src))

        # emformer attention module
        src_att, output_memory = self._apply_attention_module_forward(
            src, R, memory, attention_mask, padding_mask=padding_mask
        )
        src = src + self.dropout(src_att)

        # convolution module
        src_conv = self._apply_conv_module_forward(src, R)
        src = src + self.dropout(src_conv)

        # feed forward module
        src = src + self.dropout(self.feed_forward(src))

        src = self.norm_final(self.balancer(src))

        if alpha != 1.0:
            src = alpha * src + (1 - alpha) * src_orig

        output_utterance = src[R:]
        output_right_context = src[:R]
        return output_utterance, output_right_context, output_memory

    @torch.jit.export
    def infer(
        self,
        utterance: torch.Tensor,
        right_context: torch.Tensor,
        memory: torch.Tensor,
        attn_cache: List[torch.Tensor],
        conv_cache: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        List[torch.Tensor],
        torch.Tensor,
    ]:
        """Forward pass for inference.

         B: batch size;
         D: embedding dimension;
         R: length of right_context;
         U: length of utterance;
         M: length of memory.

        Args:
           utterance (torch.Tensor):
             Utterance frames, with shape (U, B, D).
           right_context (torch.Tensor):
             Right context frames, with shape (R, B, D).
           memory (torch.Tensor):
             Memory elements, with shape (M, B, D).
           attn_cache (List[torch.Tensor]):
             Cached attention tensors generated in preceding computation,
             including memory, key and value of left context.
           conv_cache (torch.Tensor, optional):
             Cache tensor of left context for causal convolution.
           padding_mask (torch.Tensor):
             Padding mask of ker tensor.

         Returns:
           (Tensor, Tensor, List[torch.Tensor], Tensor):
             - output utterance, with shape (U, B, D);
             - output right_context, with shape (R, B, D);
             - output memory, with shape (1, B, D) or (0, B, D).
             - output state.
             - updated conv_cache.
        """
        R = right_context.size(0)
        src = torch.cat([right_context, utterance])

        # macaron style feed forward module
        src = src + self.dropout(self.feed_forward_macaron(src))

        # emformer attention module
        (
            src_att,
            output_memory,
            attn_cache,
        ) = self._apply_attention_module_infer(
            src, R, memory, attn_cache, padding_mask=padding_mask
        )
        src = src + self.dropout(src_att)

        # convolution module
        src_conv, conv_cache = self._apply_conv_module_infer(src, R, conv_cache)
        src = src + self.dropout(src_conv)

        # feed forward module
        src = src + self.dropout(self.feed_forward(src))

        src = self.norm_final(self.balancer(src))

        output_utterance = src[R:]
        output_right_context = src[:R]
        return (
            output_utterance,
            output_right_context,
            output_memory,
            attn_cache,
            conv_cache,
        )

