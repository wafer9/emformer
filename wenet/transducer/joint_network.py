"""Transducer joint network implementation."""

import torch

class JointNetwork(torch.nn.Module):
    """Transducer joint network module.

    Args:
        joint_space_size: Dimension of joint space
        joint_activation_type: Activation type for joint network

    """

    def __init__(
        self,
        vocab_size: int,
        encoder_output_size: int,
        decoder_output_size: int,
        joint_space_size: int
    ):
        """Joint network initializer."""
        super().__init__()
        self.encoder_proj = torch.nn.Linear(encoder_output_size, joint_space_size)
        self.decoder_proj = torch.nn.Linear(decoder_output_size, joint_space_size, bias=False)
        self.output_linear = torch.nn.Linear(joint_space_size, vocab_size)

    def forward(
        self, 
        encoder_out: torch.Tensor, 
        decoder_out: torch.Tensor,
        project_input: bool = True,
    ) -> torch.Tensor:
        """Joint computation of z.

        Args:
            h_enc: Batch of expanded hidden state (B, T, 1, D_enc)
            h_dec: Batch of expanded hidden state (B, 1, U, D_dec)

        Returns:
            z: Output (B, T, U, vocab_size)

        """
        if project_input:
            logit = self.encoder_proj(encoder_out) + self.decoder_proj(decoder_out)
        else:
            logit = encoder_out + decoder_out
        logit = self.output_linear(torch.tanh(logit))

        return logit

