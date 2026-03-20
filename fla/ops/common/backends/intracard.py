"""Intra-card CP backend for shared delta rule operations.

Accelerates prefill by splitting long sequences into sub-sequences
and processing them in parallel across SMs.

Now active for:
1. Inference mode with varlen (cu_seqlens != None) - original behavior
2. Non-varlen mode with long sequences (T >= threshold) - new optimization
"""

from __future__ import annotations

import os

import torch

from fla.ops.backends import BaseBackend

# Maximum number of sub-sequences per original sequence
# Limits merge chain depth to control precision loss
MAX_SUBSEQS = int(os.environ.get('FLA_INTRACARD_MAX_SPLITS', 32))

# Minimum sequence length to enable intracard parallelism for non-varlen mode
# Below this threshold, the overhead of splitting/merging outweighs the benefit
# Benchmarking shows CP hurts at T=16384 (-9%) but helps at T>=32768 (+7-23%)
MIN_SEQ_LEN_FOR_SPLIT = int(os.environ.get('FLA_INTRACARD_MIN_SEQ_LEN', 32768))


class IntraCardCPBackend(BaseBackend):
    """Intra-card context parallel backend for chunk_gated_delta_rule_fwd_h."""

    backend_type = "intracard_cp"
    package_name = None  # No external package needed
    env_var = "FLA_INTRACARD_CP"

    @classmethod
    def is_available(cls) -> bool:
        return True

    def chunk_gated_delta_rule_fwd_h_verifier(
        self,
        k: torch.Tensor,
        w: torch.Tensor,
        u: torch.Tensor,
        g: torch.Tensor | None = None,
        gk: torch.Tensor | None = None,
        initial_state: torch.Tensor | None = None,
        output_final_state: bool = False,
        chunk_size: int = 64,
        save_new_value: bool = True,
        cu_seqlens: torch.LongTensor | None = None,
        cu_seqlens_cpu: torch.LongTensor | None = None,
        chunk_indices: torch.LongTensor | None = None,
        use_exp2: bool = False,
        transpose_state_layout: bool = False,
    ) -> tuple[bool, str | None]:
        """Check if intracard CP should handle this call."""
        # Case 1: Inference mode with varlen (original behavior)
        if torch.is_inference_mode_enabled() and cu_seqlens is not None:
            return True, None

        # Case 2: Non-varlen mode with long sequences
        # This helps both inference and forward-only training
        if cu_seqlens is None:
            B, T = k.shape[0], k.shape[1]
            # Only for batch size 1 with long sequences
            if B == 1 and T >= MIN_SEQ_LEN_FOR_SPLIT:
                return True, None

        return False, "Conditions not met for intracard CP"

    def chunk_gated_delta_rule_fwd_h(
        self,
        k: torch.Tensor,
        w: torch.Tensor,
        u: torch.Tensor,
        g: torch.Tensor | None = None,
        gk: torch.Tensor | None = None,
        initial_state: torch.Tensor | None = None,
        output_final_state: bool = False,
        chunk_size: int = 64,
        save_new_value: bool = True,
        cu_seqlens: torch.LongTensor | None = None,
        cu_seqlens_cpu: torch.LongTensor | None = None,
        chunk_indices: torch.LongTensor | None = None,
        use_exp2: bool = False,
        transpose_state_layout: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Intra-card CP implementation of chunk_gated_delta_rule_fwd_h."""
        from fla.ops.common.intracard_cp import intracard_fwd_h

        # For non-varlen mode, create cu_seqlens to enable splitting
        if cu_seqlens is None:
            B, T = k.shape[0], k.shape[1]
            # Create cu_seqlens for the single sequence
            cu_seqlens = torch.tensor([0, T], dtype=torch.long, device=k.device)
            cu_seqlens_cpu = torch.tensor([0, T], dtype=torch.long)

        return intracard_fwd_h(
            k=k, w=w, u=u, g=g, gk=gk,
            initial_state=initial_state,
            output_final_state=output_final_state,
            chunk_size=chunk_size,
            save_new_value=save_new_value,
            cu_seqlens=cu_seqlens,
            cu_seqlens_cpu=cu_seqlens_cpu,
            chunk_indices=chunk_indices,
            use_exp2=use_exp2,
            max_splits=MAX_SUBSEQS,
            transpose_state_layout=transpose_state_layout,
        )
