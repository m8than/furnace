"""Aux hidden states captured for Eagle3/DFlash draft models."""

from typing import List, Optional, Union

import torch

# Two representations coexist: models migrated to AuxHiddenStatePacker pass one
# packed [tokens, K * hidden] tensor, the rest still pass a list of K tensors.
AuxHiddenStates = Union[torch.Tensor, List[torch.Tensor]]


class AuxHiddenStatePacker:
    """Drop-in for the ``[]`` a model collects Eagle3/DFlash captures into.

    Each ``.append()`` writes into one preallocated ``[tokens, K * hidden]``
    buffer, avoiding the list path's transient ~2x HBM at ``torch.cat``.
    Assumes all captures share leading shape and feature size.
    """

    # ``append`` copies, so producers need not clone a tensor they later mutate.
    copies_on_append = True

    def __init__(
        self, num_captures: int, prototype: Optional[torch.Tensor] = None
    ) -> None:
        self._num_captures = int(num_captures)
        self._feature_size: Optional[int] = (
            int(prototype.shape[-1]) if prototype is not None else None
        )
        # Reserve large captures before layer temporaries fragment the allocator.
        # Models whose capture layout is not known yet retain lazy allocation.
        self._buffer: Optional[torch.Tensor] = (
            prototype.new_empty(
                (*prototype.shape[:-1], self._feature_size * self._num_captures)
            )
            if prototype is not None and self._num_captures > 0
            else None
        )
        self._idx = 0

    def append(self, hidden: torch.Tensor) -> None:
        # Preserve copy_'s existing casting/device-transfer behavior for callers
        # outside the direct-write path; reserve_next has a stricter contract.
        feature_size = int(hidden.shape[-1])
        if self._buffer is None:
            self._feature_size = feature_size
            self._buffer = hidden.new_empty(
                (*hidden.shape[:-1], feature_size * self._num_captures)
            )
        start = self._idx * self._feature_size
        self._buffer[..., start : start + self._feature_size].copy_(hidden)
        self._idx += 1

    def reserve_next(self, prototype: torch.Tensor) -> torch.Tensor:
        """Reserve and count one capture; the producer must fill the returned view.

        The view has the prototype's shape/dtype/device, but its row stride spans
        all captures. Like append, reservation advances len(); finalize never
        copies. Callers must finish writing the view before consuming the buffer.
        """
        if self._idx >= self._num_captures:
            raise RuntimeError("too many aux hidden state captures")
        feature_size = int(prototype.shape[-1])
        if self._buffer is None:
            self._feature_size = feature_size
            self._buffer = prototype.new_empty(
                (*prototype.shape[:-1], feature_size * self._num_captures)
            )
        start = self._idx * self._feature_size
        destination = self._buffer[..., start : start + self._feature_size]
        if (
            destination.shape != prototype.shape
            or destination.dtype != prototype.dtype
            or destination.device != prototype.device
        ):
            raise ValueError("aux hidden state capture layout changed")
        self._idx += 1
        return destination

    def __len__(self) -> int:
        return self._idx

    def finalize(self) -> torch.Tensor:
        """Return the packed buffer; callers guard the empty case on ``len()``."""
        if self._buffer is None or self._idx != self._num_captures:
            raise RuntimeError(
                f"captured {self._idx} of {self._num_captures} aux hidden states"
            )
        return self._buffer


# What a model hands down the capture path: a plain list, or a packer writing in place.
AuxHiddenStateAccumulator = Union[List[torch.Tensor], AuxHiddenStatePacker]


def pack_aux_hidden_states(aux_hidden_states: AuxHiddenStates) -> torch.Tensor:
    if isinstance(aux_hidden_states, torch.Tensor):
        return aux_hidden_states
    if len(aux_hidden_states) == 1:
        return aux_hidden_states[0]
    return torch.cat(aux_hidden_states, dim=-1)
