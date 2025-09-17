from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional
from scipy.sparse import csr_matrix

@dataclass
class InteractionDataset:
    user_map: Dict[str, int]
    item_map: Dict[str, int]
    R: csr_matrix                     # [U, I] ratings or implicit counts
    timestamps: Optional[csr_matrix] = None

    @property
    def n_users(self) -> int: return self.R.shape[0]
    @property
    def n_items(self) -> int: return self.R.shape[1]
