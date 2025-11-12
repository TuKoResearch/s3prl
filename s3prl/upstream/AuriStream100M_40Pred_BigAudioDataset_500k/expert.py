from collections import OrderedDict
from typing import Dict, List, Union

import torch.nn as nn
from torch import Tensor
import torch
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F

from transformers import AutoConfig, AutoModel

SAMPLE_RATE = 16000

class UpstreamExpert(nn.Module):
    def __init__(self, name: str = '', **kwargs):
        super().__init__()
        self.name = "AuriStream100M_40Pred_BigAudioDataset_500k"

        # Quantizer and LM
        self.extracter = AutoModel.from_pretrained(
            "TuKoResearch/WavCochCausalV8192", trust_remote_code=True
        )
        self.model = AutoModel.from_pretrained(
            "TuKoResearch/AuriStream100M_40Pred_BigAudioDataset_500k",
            trust_remote_code=True
        )
        self.config = AutoConfig.from_pretrained(
            "TuKoResearch/AuriStream100M_40Pred_BigAudioDataset_500k",
            trust_remote_code=True
        )

        # 5 ms / token at 16 kHz -> 80 samples per token
        self.hop = int(SAMPLE_RATE * 0.005)        # 80
        self.max_tokens = 4096
        self.max_samples = self.hop * self.max_tokens  # 327_680

    def get_downsample_rates(self, key: str = None) -> int:
        # S3PRL uses this (samples per frame/token)
        return  self.hop

    def forward(self, wavs: List[Tensor]) -> Dict[str, Union[Tensor, List[Tensor]]]:
        """
        Return a dict whose values are lists of (B, T, D) tensors (layers).
        - Trim inputs to <= 4096 tokens (327,680 samples).
        - If an input was trimmed, pad the model outputs with trailing zeros to
          match the original expected #tokens (orig_len // hop).
        """
        # Keep original per-utterance lengths (in samples) BEFORE any padding/cropping
        device = wavs[0].device
        raw = [w if w.ndim == 1 else w.squeeze() for w in wavs]
        orig_lens = torch.tensor([r.numel() for r in raw], device=device, dtype=torch.long)
        # Expected tokens from original audio (integer floor)
        exp_tokens = orig_lens // self.hop
        T_out_max = int(exp_tokens.max().item()) if exp_tokens.numel() > 0 else 0

        # Batch pad to max original length, then crop the *processed* region to <= 327,680 samples
        padded = pad_sequence(raw, batch_first=True)                     # [B, T_pad]
        T_proc_samples = min(padded.size(1), self.max_samples)           # cap at 327,680
        padded = padded[:, :T_proc_samples].unsqueeze(1)                 # [B, 1, T_proc]

        # Ensure modules and tensors live on the same device
        self.extracter = self.extracter.to(device)
        self.model = self.model.to(device)
        padded = padded.to(device)

        # Wavcoch -> token IDs (B, T_tok) in [0, 8191]
        enc = self.extracter(padded, sample_rate=SAMPLE_RATE)
        enc = enc.to(device) if hasattr(enc, "to") else enc
        ids = getattr(enc, "input_ids", None)
        if ids is None:
            # Older wrappers may use 'input_values' to hold integer token IDs
            ids = enc.input_values
        # Safety clamp to 4096 tokens
        ids = ids[:, : self.max_tokens]

        # AuriStream -> hidden states (list of layers), shapes: (B, T_proc_tokens, D)
        outputs = self.model(ids, output_hidden_states=True, return_dict=True, normalize_embeddings='learned')
        hs_list = list(outputs.hidden_states)

        # Build per-layer outputs, zero-padded so that for each utterance i:
        # - if not trimmed: length == exp_tokens[i]
        # - if trimmed (exp_tokens[i] > 4096): length == exp_tokens[i], with zeros after 4096
        B = ids.size(0)
        T_proc_tokens = hs_list[-1].size(1) if len(hs_list) else 0
        D = hs_list[-1].size(-1) if len(hs_list) else 0
        T_batch_out = T_out_max if T_out_max > 0 else T_proc_tokens

        padded_layers: List[Tensor] = []
        for h in hs_list:
            # Allocate zeros for the full (batch, max_expected_tokens, dim)
            out_h = h.new_zeros((B, T_batch_out, D))
            for i in range(B):
                # Use only tokens that correspond to actual (non-padded) audio,
                # capped by what we really processed (<= 4096)
                t_i = int(min(exp_tokens[i].item(), T_proc_tokens))
                if t_i > 0:
                    out_h[i, :t_i, :] = h[i, :t_i, :]
                # If exp_tokens[i] > T_proc_tokens (trimmed), the tail stays zeros by design.
            padded_layers.append(out_h)

        # Return hidden states
        return {"hidden_states": padded_layers}
