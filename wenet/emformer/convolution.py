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


class ConvolutionModule(nn.Module):
    """ConvolutionModule.

    Modified from https://github.com/pytorch/audio/blob/main/torchaudio/prototype/models/conv_emformer.py # noqa

    Args:
      chunk_length (int):
        Length of each chunk.
      right_context_length (int):
        Length of right context.
      channels (int):
        The number of input channels and output channels of conv layers.
      kernel_size (int):
        Kernerl size of conv layers.
      bias (bool):
        Whether to use bias in conv layers (default=True).
    """

    def __init__(
        self,
        chunk_length: int,
        right_context_length: int,
        channels: int,
        kernel_size: int,
        bias: bool = True,
    ) -> None:
        """Construct an ConvolutionModule object."""
        super().__init__()
        # kernerl_size should be an odd number for 'SAME' padding
        assert (kernel_size - 1) % 2 == 0, kernel_size

        self.chunk_length = chunk_length
        self.right_context_length = right_context_length
        self.channels = channels

        self.pointwise_conv1 = ScaledConv1d(
            channels,
            2 * channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=bias,
        )
        # After pointwise_conv1 we put x through a gated linear unit
        # (nn.functional.glu).
        # For most layers the normal rms value of channels of x seems to be in
        # the range 1 to 4, but sometimes, for some reason, for layer 0 the rms
        # ends up being very large, between 50 and 100 for different channels.
        # This will cause very peaky and sparse derivatives for the sigmoid
        # gating function, which will tend to make the loss function not learn
        # effectively.  (for most layers the average absolute values are in the
        # range 0.5..9.0, and the average p(x>0), i.e. positive proportion,
        # at the output of pointwise_conv1.output is around 0.35 to 0.45 for
        # different layers, which likely breaks down as 0.5 for the "linear"
        # half and 0.2 to 0.3 for the part that goes into the sigmoid.
        # The idea is that if we constrain the rms values to a reasonable range
        # via a constraint of max_abs=10.0, it will be in a better position to
        # start learning something, i.e. to latch onto the correct range.
        self.deriv_balancer1 = ActivationBalancer(
            channel_dim=1, max_abs=10.0, min_positive=0.05, max_positive=1.0
        )

        # make it causal by padding cached (kernel_size - 1) frames on the left
        self.cache_size = kernel_size - 1
        self.depthwise_conv = ScaledConv1d(
            channels,
            channels,
            kernel_size,
            stride=1,
            padding=0,
            groups=channels,
            bias=bias,
        )

        self.deriv_balancer2 = ActivationBalancer(
            channel_dim=1, min_positive=0.05, max_positive=1.0
        )

        self.activation = DoubleSwish()

        self.pointwise_conv2 = ScaledConv1d(
            channels,
            channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=bias,
            initial_scale=0.25,
        )

    def _split_right_context(
        self,
        pad_utterance: torch.Tensor,
        right_context: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
          pad_utterance:
            Its shape is (cache_size + U, B, D).
          right_context:
            Its shape is (R, B, D).

        Returns:
          Right context segments padding with corresponding context.
          Its shape is (num_segs * B, D, cache_size + right_context_length).
        """
        U_, B, D = pad_utterance.size()
        R = right_context.size(0)
        assert self.right_context_length != 0
        assert R % self.right_context_length == 0
        num_chunks = R // self.right_context_length
        right_context = right_context.reshape(
            num_chunks, self.right_context_length, B, D
        )
        right_context = right_context.permute(0, 2, 1, 3).reshape(
            num_chunks * B, self.right_context_length, D
        )

        intervals = torch.arange(
            0, self.chunk_length * (num_chunks - 1), self.chunk_length
        )
        first = torch.arange(
            self.chunk_length, self.chunk_length + self.cache_size
        )
        indexes = intervals.unsqueeze(1) + first.unsqueeze(0)
        indexes = torch.cat(
            [indexes, torch.arange(U_ - self.cache_size, U_).unsqueeze(0)]
        )
        padding = pad_utterance[indexes]  # (num_chunks, cache_size, B, D)
        padding = padding.permute(0, 2, 1, 3).reshape(
            num_chunks * B, self.cache_size, D
        )

        pad_right_context = torch.cat([padding, right_context], dim=1)
        # (num_chunks * B, cache_size + right_context_length, D)
        return pad_right_context.permute(0, 2, 1)

    def _merge_right_context(
        self, right_context: torch.Tensor, B: int
    ) -> torch.Tensor:
        """
        Args:
          right_context:
            Right context segments.
            It shape is (num_segs * B, D, right_context_length).
          B:
            Batch size.

        Returns:
          A tensor of shape (B, D, R), where
          R = num_segs * right_context_length.
        """
        right_context = right_context.reshape(
            -1, B, self.channels, self.right_context_length
        )
        right_context = right_context.permute(1, 2, 0, 3)
        right_context = right_context.reshape(B, self.channels, -1)
        return right_context

    def forward(
        self,
        utterance: torch.Tensor,
        right_context: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Causal convolution module.

        Args:
          utterance (torch.Tensor):
            Utterance tensor of shape (U, B, D).
          right_context (torch.Tensor):
            Right context tensor of shape (R, B, D).

        Returns:
          A tuple of 2 tensors:
          - output utterance of shape (U, B, D).
          - output right_context of shape (R, B, D).
        """
        U, B, D = utterance.size()
        R, _, _ = right_context.size()

        # point-wise conv and GLU mechanism
        x = torch.cat([right_context, utterance], dim=0)  # (R + U, B, D)
        x = x.permute(1, 2, 0)  # (B, D, R + U)
        x = self.pointwise_conv1(x)  # (B, 2 * D, R + U)
        x = self.deriv_balancer1(x)
        x = nn.functional.glu(x, dim=1)  # (B, D, R + U)
        utterance = x[:, :, R:]  # (B, D, U)
        right_context = x[:, :, :R]  # (B, D, R)

        # make causal convolution
        cache = torch.zeros(
            B, D, self.cache_size, device=x.device, dtype=x.dtype
        )
        pad_utterance = torch.cat(
            [cache, utterance], dim=2
        )  # (B, D, cache + U)

        # depth-wise conv on utterance
        utterance = self.depthwise_conv(pad_utterance)  # (B, D, U)

        if self.right_context_length > 0:
            # depth-wise conv on right_context
            pad_right_context = self._split_right_context(
                pad_utterance.permute(2, 0, 1), right_context.permute(2, 0, 1)
            )  # (num_segs * B, D, cache_size + right_context_length)
            right_context = self.depthwise_conv(
                pad_right_context
            )  # (num_segs * B, D, right_context_length)
            right_context = self._merge_right_context(
                right_context, B
            )  # (B, D, R)

        x = torch.cat([right_context, utterance], dim=2)  # (B, D, R + U)
        x = self.deriv_balancer2(x)
        x = self.activation(x)

        # point-wise conv
        x = self.pointwise_conv2(x)  # (B, D, R + U)

        right_context = x[:, :, :R]  # (B, D, R)
        utterance = x[:, :, R:]  # (B, D, U)
        return (
            utterance.permute(2, 0, 1),
            right_context.permute(2, 0, 1),
        )

    @torch.jit.export
    def infer(
        self,
        utterance: torch.Tensor,
        right_context: torch.Tensor,
        cache: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Causal convolution module applied on both utterance and right_context.

        Args:
          utterance (torch.Tensor):
            Utterance tensor of shape (U, B, D).
          right_context (torch.Tensor):
            Right context tensor of shape (R, B, D).
          cache (torch.Tensor, optional):
            Cached tensor for left padding of shape (B, D, cache_size).

        Returns:
          A tuple of 3 tensors:
            - output utterance of shape (U, B, D).
            - output right_context of shape (R, B, D).
            - updated cache tensor of shape (B, D, cache_size).
        """
        U, B, D = utterance.size()
        R, _, _ = right_context.size()

        # point-wise conv
        x = torch.cat([utterance, right_context], dim=0)  # (U + R, B, D)
        x = x.permute(1, 2, 0)  # (B, D, U + R)
        x = self.pointwise_conv1(x)  # (B, 2 * D, U + R)
        x = self.deriv_balancer1(x)
        x = nn.functional.glu(x, dim=1)  # (B, D, U + R)

        # make causal convolution
        assert cache.shape == (B, D, self.cache_size), cache.shape
        x = torch.cat([cache, x], dim=2)  # (B, D, cache_size + U + R)
        # update cache
        new_cache = x[:, :, -R - self.cache_size : -R]

        # 1-D depth-wise conv
        x = self.depthwise_conv(x)  # (B, D, U + R)

        x = self.deriv_balancer2(x)
        x = self.activation(x)

        # point-wise conv
        x = self.pointwise_conv2(x)  # (B, D, U + R)

        utterance = x[:, :, :U]  # (B, D, U)
        right_context = x[:, :, U:]  # (B, D, R)
        return (
            utterance.permute(2, 0, 1),
            right_context.permute(2, 0, 1),
            new_cache,
        )

