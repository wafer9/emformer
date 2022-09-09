from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from wenet.emformer.scaling import (
    ActivationBalancer,
    BasicNorm,
    DoubleSwish,
    ScaledConv1d,
    ScaledConv2d,
    ScaledLinear,
)


class EmformerAttention(nn.Module):
    """Emformer layer attention module.

    Args:
      embed_dim (int):
        Embedding dimension.
      nhead (int):
        Number of attention heads in each Emformer layer.
      dropout (float, optional):
        Dropout probability. (Default: 0.0)
      tanh_on_mem (bool, optional):
        If ``True``, applies tanh to memory elements. (Default: ``False``)
      negative_inf (float, optional):
        Value to use for negative infinity in attention weights. (Default: -1e8)
    """

    def __init__(
        self,
        embed_dim: int,
        nhead: int,
        dropout: float = 0.0,
        tanh_on_mem: bool = False,
        negative_inf: float = -1e8,
    ):
        super().__init__()

        if embed_dim % nhead != 0:
            raise ValueError(
                f"embed_dim ({embed_dim}) is not a multiple of"
                f"nhead ({nhead})."
            )

        self.embed_dim = embed_dim
        self.nhead = nhead
        self.tanh_on_mem = tanh_on_mem
        self.negative_inf = negative_inf
        self.head_dim = embed_dim // nhead
        self.dropout = dropout

        self.emb_to_key_value = ScaledLinear(
            embed_dim, 2 * embed_dim, bias=True
        )
        self.emb_to_query = ScaledLinear(embed_dim, embed_dim, bias=True)
        self.out_proj = ScaledLinear(
            embed_dim, embed_dim, bias=True, initial_scale=0.25
        )

    def _gen_attention_probs(
        self,
        attention_weights: torch.Tensor,
        attention_mask: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Given the entire attention weights, mask out unecessary connections
        and optionally with padding positions, to obtain underlying chunk-wise
        attention probabilities.

        B: batch size;
        Q: length of query;
        KV: length of key and value.

        Args:
          attention_weights (torch.Tensor):
            Attention weights computed on the entire concatenated tensor
            with shape (B * nhead, Q, KV).
          attention_mask (torch.Tensor):
            Mask tensor where chunk-wise connections are filled with `False`,
            and other unnecessary connections are filled with `True`,
            with shape (Q, KV).
          padding_mask (torch.Tensor, optional):
            Mask tensor where the padding positions are fill with `True`,
            and other positions are filled with `False`, with shapa `(B, KV)`.

        Returns:
          A tensor of shape (B * nhead, Q, KV).
        """
        attention_weights_float = attention_weights.float()
        attention_weights_float = attention_weights_float.masked_fill(
            attention_mask.unsqueeze(0), self.negative_inf
        )
        if padding_mask is not None:
            Q = attention_weights.size(1)
            B = attention_weights.size(0) // self.nhead
            attention_weights_float = attention_weights_float.view(
                B, self.nhead, Q, -1
            )
            attention_weights_float = attention_weights_float.masked_fill(
                padding_mask.unsqueeze(1).unsqueeze(2).to(torch.bool),
                self.negative_inf,
            )
            attention_weights_float = attention_weights_float.view(
                B * self.nhead, Q, -1
            )

        attention_probs = nn.functional.softmax(
            attention_weights_float, dim=-1
        ).type_as(attention_weights)

        attention_probs = nn.functional.dropout(
            attention_probs, p=self.dropout, training=self.training
        )
        return attention_probs

    def _forward_impl(
        self,
        utterance: torch.Tensor,
        right_context: torch.Tensor,
        summary: torch.Tensor,
        memory: torch.Tensor,
        attention_mask: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        left_context_key: Optional[torch.Tensor] = None,
        left_context_val: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Underlying chunk-wise attention implementation."""
        U, B, _ = utterance.size()
        R = right_context.size(0)
        M = memory.size(0)
        scaling = float(self.head_dim) ** -0.5

        # compute query with [right_context, utterance, summary].
        query = self.emb_to_query(
            torch.cat([right_context, utterance, summary])
        )
        # compute key and value with [memory, right_context, utterance].
        key, value = self.emb_to_key_value(
            torch.cat([memory, right_context, utterance])
        ).chunk(chunks=2, dim=2)

        if left_context_key is not None and left_context_val is not None:
            # now compute key and value with
            #   [memory, right context, left context, uttrance]
            # this is used in inference mode
            key = torch.cat([key[: M + R], left_context_key, key[M + R :]])
            value = torch.cat(
                [value[: M + R], left_context_val, value[M + R :]]
            )
        Q = query.size(0)
        # KV = key.size(0)

        reshaped_query, reshaped_key, reshaped_value = [
            tensor.contiguous()
            .view(-1, B * self.nhead, self.head_dim)
            .transpose(0, 1)
            for tensor in [query, key, value]
        ]  # (B * nhead, Q or KV, head_dim)
        attention_weights = torch.bmm(
            reshaped_query * scaling, reshaped_key.transpose(1, 2)
        )  # (B * nhead, Q, KV)

        # compute attention probabilities
        attention_probs = self._gen_attention_probs(
            attention_weights, attention_mask, padding_mask
        )

        # compute attention outputs
        attention = torch.bmm(attention_probs, reshaped_value)
        assert attention.shape == (B * self.nhead, Q, self.head_dim)
        attention = (
            attention.transpose(0, 1).contiguous().view(Q, B, self.embed_dim)
        )

        # apply output projection
        outputs = self.out_proj(attention)

        output_right_context_utterance = outputs[: R + U]
        output_memory = outputs[R + U :]
        if self.tanh_on_mem:
            output_memory = torch.tanh(output_memory)
        else:
            output_memory = torch.clamp(output_memory, min=-10, max=10)

        return output_right_context_utterance, output_memory, key, value

    def forward(
        self,
        utterance: torch.Tensor,
        right_context: torch.Tensor,
        summary: torch.Tensor,
        memory: torch.Tensor,
        attention_mask: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # TODO: Modify docs.
        """Forward pass for training and validation mode.

        B: batch size;
        D: embedding dimension;
        R: length of the hard-copied right contexts;
        U: length of full utterance;
        S: length of summary vectors;
        M: length of memory vectors.

        It computes a `big` attention matrix on full utterance and
        then utilizes a pre-computed mask to simulate chunk-wise attention.

        It concatenates three blocks: hard-copied right contexts,
        full utterance, and summary vectors, as a `big` block,
        to compute the query tensor:
        query = [right_context, utterance, summary],
        with length Q = R + U + S.
        It concatenates the three blocks: memory vectors,
        hard-copied right contexts, and full utterance as another `big` block,
        to compute the key and value tensors:
        key & value = [memory, right_context, utterance],
        with length KV = M + R + U.
        Attention scores is computed with above `big` query and key.

        Then the underlying chunk-wise attention is obtained by applying
        the attention mask. Suppose
        c_i: chunk at index i;
        r_i: right context that c_i can use;
        l_i: left context that c_i can use;
        m_i: past memory vectors from previous layer that c_i can use;
        s_i: summary vector of c_i;
        The target chunk-wise attention is:
        c_i, r_i (in query) -> l_i, c_i, r_i, m_i (in key);
        s_i (in query) -> l_i, c_i, r_i (in key).

        Args:
          utterance (torch.Tensor):
            Full utterance frames, with shape (U, B, D).
          right_context (torch.Tensor):
            Hard-copied right context frames, with shape (R, B, D),
            where R = num_chunks * right_context_length
          summary (torch.Tensor):
            Summary elements with shape (S, B, D), where S = num_chunks.
            It is an empty tensor without using memory.
          memory (torch.Tensor):
            Memory elements, with shape (M, B, D), where M = num_chunks - 1.
            It is an empty tensor without using memory.
          attention_mask (torch.Tensor):
            Pre-computed attention mask to simulate underlying chunk-wise
            attention, with shape (Q, KV).
          padding_mask (torch.Tensor):
            Padding mask of key tensor, with shape (B, KV).

        Returns:
          A tuple containing 2 tensors:
            - output of right context and utterance, with shape (R + U, B, D).
            - memory output, with shape (M, B, D), where M = S - 1 or M = 0.
        """
        (
            output_right_context_utterance,
            output_memory,
            _,
            _,
        ) = self._forward_impl(
            utterance,
            right_context,
            summary,
            memory,
            attention_mask,
            padding_mask=padding_mask,
        )
        return output_right_context_utterance, output_memory[:-1]

    @torch.jit.export
    def infer(
        self,
        utterance: torch.Tensor,
        right_context: torch.Tensor,
        summary: torch.Tensor,
        memory: torch.Tensor,
        left_context_key: torch.Tensor,
        left_context_val: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass for inference.

        B: batch size;
        D: embedding dimension;
        R: length of right context;
        U: length of utterance, i.e., current chunk;
        L: length of cached left context;
        S: length of summary vectors, S = 1;
        M: length of cached memory vectors.

        It concatenates the right context, utterance (i.e., current chunk)
        and summary vector of current chunk, to compute the query tensor:
        query = [right_context, utterance, summary],
        with length Q = R + U + S.
        It concatenates the memory vectors, right context, left context, and
        current chunk, to compute the key and value tensors:
        key & value = [memory, right_context, left_context, utterance],
        with length KV = M + R + L + U.

        The chunk-wise attention is:
        chunk, right context (in query) ->
          left context, chunk, right context, memory vectors (in key);
        summary (in query) -> left context, chunk, right context (in key).

        Args:
          utterance (torch.Tensor):
            Current chunk frames, with shape (U, B, D), where U = chunk_length.
          right_context (torch.Tensor):
            Right context frames, with shape (R, B, D),
            where R = right_context_length.
          summary (torch.Tensor):
            Summary vector with shape (1, B, D), or empty tensor.
          memory (torch.Tensor):
            Memory vectors, with shape (M, B, D), or empty tensor.
          left_context_key (torch,Tensor):
            Cached attention key of left context from preceding computation,
            with shape (L, B, D).
          left_context_val (torch.Tensor):
            Cached attention value of left context from preceding computation,
            with shape (L, B, D).
          padding_mask (torch.Tensor):
            Padding mask of key tensor, with shape (B, KV).

        Returns:
          A tuple containing 4 tensors:
            - output of right context and utterance, with shape (R + U, B, D).
            - memory output, with shape (1, B, D) or (0, B, D).
            - attention key of left context and utterance, which would be cached
              for next computation, with shape (L + U, B, D).
            - attention value of left context and utterance, which would be
              cached for next computation, with shape (L + U, B, D).
        """
        U = utterance.size(0)
        R = right_context.size(0)
        L = left_context_key.size(0)
        S = summary.size(0)
        M = memory.size(0)

        # TODO: move it outside
        # query = [right context, utterance, summary]
        Q = R + U + S
        # key, value = [memory, right context, left context, uttrance]
        KV = M + R + L + U
        attention_mask = torch.zeros(Q, KV).to(
            dtype=torch.bool, device=utterance.device
        )
        # disallow attention bettween the summary vector with the memory bank
        attention_mask[-1, :M] = True
        (
            output_right_context_utterance,
            output_memory,
            key,
            value,
        ) = self._forward_impl(
            utterance,
            right_context,
            summary,
            memory,
            attention_mask,
            padding_mask=padding_mask,
            left_context_key=left_context_key,
            left_context_val=left_context_val,
        )
        return (
            output_right_context_utterance,
            output_memory,
            key[M + R :],
            value[M + R :],
        )
