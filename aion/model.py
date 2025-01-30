import torch
from typing import Dict, Tuple, Any, Optional

from .fourm.fm import FM
from .fourm.modality_info import MODALITY_INFO


class AION(FM):
    """
    Wrapper for 4M model including additional utilities.
    """

    def embed_inputs(
        self,
        input_dict: Dict[str, torch.Tensor],
        mask: Optional[Dict[str, torch.Tensor]] = None,
        num_encoder_tokens: int = 256,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Builds the encoder modality dictionary given some input data.
        Optionally, if mask is provided, input token masking can be used.

        Args:
            X (Dict[str, torch.Tensor]): Input data dictionary.
            mask (Dict[str, torch.Tensor], optional): Mask dictionary. Defaults to {}.
            num_encoder_tokens (int, optional): Maximum number of encoder tokens. Defaults to 256.

        Returns:
            tuple:
                - encoder_tokens (torch.Tensor): Selected encoder tokens from all modalities. Shape (B, N, D) where N is the number of selected encoder tokens.
                - encoder_emb (torch.Tensor): Corresponding embeddings for encoder tokens. Shape (B, N, D)
                - encoder_mask (torch.Tensor): A boolean mask indicating which encoder tokens are valid (set to 0 for valid tokens, 1 otherwise). Shape (B, 1, N)
                - mod_mask (torch.Tensor): An integer mask marking the modality type for each encoder token (with -1 indicating unassigned pad tokens). Shape (B, N)
        """
        if mask is None:
            mask = {}
        assert isinstance(input_dict, dict), "first input must be a dictionary"
        assert isinstance(mask, dict), "Mask must be a dictionary if provided"
        assert all(
            key in input_dict for key in mask
        ), "All keys in the input mask must be in X"
        assert all(
            key in self.encoder_embeddings for key in input_dict.keys()
        ), "All keys in X must be in self.encoder_embeddings"

        device = next(self.parameters()).device

        encoder_mod_dict = {}
        for mod, tensor in input_dict.items():
            tensor = tensor.to(torch.long).to(device)
            if tensor.dim() == 1:
                tensor = tensor.unsqueeze(1)
            input_mask = mask.get(
                mod,
                torch.zeros(
                    tensor.shape[0], tensor.shape[1], dtype=torch.bool, device=device
                ),
            )
            if MODALITY_INFO[mod]["type"] == "img":
                assert (
                    tensor.shape[1] == self.encoder_embeddings[mod].num_patches
                ), f"Expected size {self.encoder_embeddings[mod].num_patches} for modality {mod}, but got {tensor.shape[1]}"

            encoder_mod_dict[mod] = self.encoder_embeddings[mod](
                {"tensor": tensor, "input_mask": input_mask}
            )

        encoder_tokens, encoder_emb, encoder_mask, encoder_mod_mask = (
            self.forward_mask_encoder(encoder_mod_dict, num_encoder_tokens)
        )

        return encoder_tokens, encoder_emb, encoder_mask, encoder_mod_mask

    def embed_targets(
        self,
        target_mask: Optional[Dict[str, torch.Tensor]] = None,
        target_dict: Optional[Dict[str, torch.Tensor]] = None,
        num_decoder_tokens: int = 256,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """

        Returns:
            tuple:
                - decoder_tokens (torch.Tensor): Selected decoder tokens from all modalities. Shape (B, M, D) where M is the number of selected decoder tokens.
                - decoder_emb (torch.Tensor): Corresponding embeddings for decoder tokens. Shape (B, M, D)
                - decoder_mask (torch.Tensor): A boolean mask indicating which decoder tokens are valid (set to 0 for valid tokens, 1 otherwise). Shape (B, 1, M)
                - target_ids (torch.Tensor): IDs of the target tokens corresponding to the decoder tokens. Shape (B, M)
                - decoder_attention_mask (torch.Tensor): Mask for the decoder self-attention layers. Shape (B, M, M)
                - mod_mask (torch.Tensor): An integer mask marking the modality type for each decoder token (with -1 indicating unassigned pad tokens). Shape (B, M)
        """
        assert isinstance(target_mask, dict), "Traget mask must be a dictionary"
        assert all(
            key in self.decoder_embeddings for key in target_mask.keys()
        ), "All keys in target mask must be in self.decoder_embeddings"

        device = next(self.parameters()).device

        if target_mask is not None:
            decoder_mod_dict = {}
            for mod, mask in target_mask.items():
                mask = mask.to(torch.bool).to(device)
                tensor = torch.zeros_like(mask).to(torch.long).to(device)
                decoder_attention_mask = (
                    torch.zeros_like(mask).to(torch.bool).to(device)
                )
                decoder_mod_dict[mod] = self.decoder_embeddings[mod].forward_embed(
                    {
                        "tensor": tensor,
                        "target_mask": mask,
                        "decoder_attention_mask": decoder_attention_mask,
                    }
                )
        elif target_dict is not None:
            decoder_mod_dict = {
                mod: self.decoder_embeddings[mod].forward_embed(d)
                for mod, d in target_dict.items()
                if mod in self.decoder_embeddings
            }
        else:
            raise ValueError("Either target_mask or target_dict must be provided")

        (
            decoder_tokens,
            decoder_emb,
            decoder_mask,
            target_ids,
            decoder_attention_mask,
            decoder_mod_mask,
        ) = self.forward_mask_decoder(decoder_mod_dict, num_decoder_tokens)
        return (
            decoder_tokens,
            decoder_emb,
            decoder_mask,
            target_ids,
            decoder_attention_mask,
            decoder_mod_mask,
        )

    def _encode(self, encoder_tokens, encoder_emb, encoder_mask):
        x = encoder_tokens + encoder_emb
        x = self.forward_encoder(x, encoder_mask=encoder_mask)
        context = self.decoder_proj_context(x) + encoder_emb
        return context

    def _decode(
        self,
        encoder_outputs,
        encoder_mask,
        decoder_tokens,
        decoder_emb,
        decoder_attention_mask,
    ):
        x = decoder_tokens + decoder_emb
        x = self.forward_decoder(
            x,
            encoder_outputs,
            encoder_mask=encoder_mask,
            decoder_attention_mask=decoder_attention_mask,
        )
        return x

    def encode(
        self,
        input_dict: Dict[str, torch.Tensor],
        input_mask: Optional[Dict[str, torch.Tensor]] = None,
        num_encoder_tokens: int = 256,
    ) -> torch.Tensor:
        """
        Encode input data using the mode

        Args:
            num_encoder_tokens (int, optional): Maximum number of encoder tokens. Defaults to 256.
        """
        encoder_tokens, encoder_emb, encoder_mask, _ = self.embed_inputs(
            input_dict, mask=input_mask, num_encoder_tokens=num_encoder_tokens
        )
        return self._encode(encoder_tokens, encoder_emb, encoder_mask)

    def forward(
        self,
        input_dict: Dict[str, torch.Tensor],
        target_mask: Optional[Dict[str, torch.Tensor]] = None,
        target_dict: Optional[Dict[str, torch.Tensor]] = None,
        input_mask: Optional[Dict[str, torch.Tensor]] = None,
        num_decoder_tokens: int = 256,
        num_encoder_tokens: int = 256,
    ) -> torch.Tensor:
        """
        The forward function returns the logits of the requested target outputs, given the input data.

        Args:
            input_dict (Dict[str, torch.Tensor]): Input data dictionary.
            target_mask (Dict[str, torch.Tensor]): Target mask dictionary, defines which modalities to predict, and which tokens within that modality.
            input_mask (Dict[str, torch.Tensor], optional): Input mask dictionary. Defaults to None.
            num_encoder_tokens (int, optional): Maximum number of encoder tokens. Defaults to 256.

        Returns:
            torch.Tensor: Output tensor of the model.
        """
        if target_mask is None and target_dict is None:
            raise ValueError("Either target_mask or target_dict must be provided")
        if target_mask is not None and target_dict is not None:
            raise ValueError("Only one of target_mask or target_dict can be provided")

        # Embedding inputs and targets
        encoder_tokens, encoder_emb, encoder_mask, _ = self.embed_inputs(
            input_dict, mask=input_mask, num_encoder_tokens=num_encoder_tokens
        )
        (
            decoder_tokens,
            decoder_emb,
            decoder_mask,
            target_ids,
            decoder_attention_mask,
            decoder_mod_mask,
        ) = self.embed_targets(
            target_mask, target_dict, num_decoder_tokens=num_decoder_tokens
        )

        # Run the encoder
        encoder_output = self._encode(encoder_tokens, encoder_emb, encoder_mask)
        decoder_output = self._decode(
            encoder_output,
            encoder_mask,
            decoder_tokens,
            decoder_emb,
            decoder_attention_mask,
        )

        # Now, we compute the logits for the requested tokens and return them
        mod_logits = {}
        for mod in target_mask.keys():
            idx = self.modality_info[mod]["id"]
            mod_logits[mod] = self.decoder_embeddings[mod].forward_logits(
                decoder_output[decoder_mod_mask == idx]
            )

        return mod_logits
