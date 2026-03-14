from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
from torch import nn
from torch_geometric.nn import GATConv, global_mean_pool


@dataclass
class PairTripletBatch:
    pair_indices: torch.Tensor  # [2, P]
    pair_dist: torch.Tensor  # [P, 1]
    pair_bond_flag: torch.Tensor  # [P, 1]
    pair_graph: torch.Tensor  # [P]
    pair_vu_idx: torch.Tensor  # [T]
    pair_uw_idx: torch.Tensor  # [T]
    pair_vw_idx: torch.Tensor  # [T]
    geom_features: torch.Tensor  # [T, 4]


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Local2FWLUpdate(nn.Module):
    def __init__(self, pair_dim: int, geom_dim: int = 4):
        super().__init__()
        self.psi = MLP(in_dim=3 * pair_dim + geom_dim, hidden_dim=pair_dim, out_dim=pair_dim)
        self.phi = MLP(in_dim=2 * pair_dim, hidden_dim=pair_dim, out_dim=pair_dim)

    def forward(
        self,
        h_pair: torch.Tensor,
        pair_vu_idx: torch.Tensor,
        pair_uw_idx: torch.Tensor,
        pair_vw_idx: torch.Tensor,
        geom_features: torch.Tensor,
    ) -> torch.Tensor:
        if pair_vw_idx.numel() == 0:
            return h_pair

        h_vu = h_pair[pair_vu_idx]
        h_uw = h_pair[pair_uw_idx]
        h_vw = h_pair[pair_vw_idx]

        triplet_input = torch.cat([h_vu, h_uw, h_vw, geom_features], dim=-1)
        m_triplet = self.psi(triplet_input)

        agg = torch.zeros_like(h_pair)
        agg.index_add_(0, pair_vw_idx, m_triplet)

        updated = self.phi(torch.cat([h_pair, agg], dim=-1))
        return h_pair + updated


class Local2FWLGAT(nn.Module):
    def __init__(
        self,
        node_in_dim: int,
        hidden_dim: int,
        pair_layers: int = 3,
        gat_layers: int = 2,
        gat_heads: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.node_encoder = MLP(node_in_dim, hidden_dim, hidden_dim)
        self.pair_init = MLP(2 * hidden_dim + 2, hidden_dim, hidden_dim)

        self.local2fwl_layers = nn.ModuleList([Local2FWLUpdate(hidden_dim) for _ in range(pair_layers)])
        self.node_fuse = MLP(2 * hidden_dim, hidden_dim, hidden_dim)

        self.gat_layers = nn.ModuleList()
        for _ in range(gat_layers):
            self.gat_layers.append(
                GATConv(
                    in_channels=hidden_dim,
                    out_channels=hidden_dim // gat_heads,
                    heads=gat_heads,
                    concat=True,
                    dropout=dropout,
                )
            )

        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    @staticmethod
    def _pair_key(a: int, b: int) -> Tuple[int, int]:
        return (a, b) if a < b else (b, a)

    @staticmethod
    def _angle(v1: torch.Tensor, v2: torch.Tensor) -> torch.Tensor:
        denom = (v1.norm(dim=-1) * v2.norm(dim=-1)).clamp_min(1e-9)
        cos = (v1 * v2).sum(dim=-1) / denom
        cos = cos.clamp(-1.0, 1.0)
        return torch.acos(cos)

    def _build_pair_triplet_structures(self, pos: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor) -> PairTripletBatch:
        device = pos.device
        num_nodes = pos.size(0)
        num_graphs = int(batch.max().item()) + 1 if batch.numel() > 0 else 0

        graph_nodes: List[torch.Tensor] = [(batch == g).nonzero(as_tuple=False).view(-1) for g in range(num_graphs)]

        bonded_pairs: set[Tuple[int, int]] = set()
        for k in range(edge_index.size(1)):
            a = int(edge_index[0, k].item())
            b = int(edge_index[1, k].item())
            if a == b:
                continue
            bonded_pairs.add(self._pair_key(a, b))

        active_pairs = set(bonded_pairs)
        triplets: List[Tuple[int, int, int]] = []

        for g_nodes in graph_nodes:
            n = g_nodes.numel()
            if n < 3:
                continue

            local_pos = pos[g_nodes]  # [n, 3]
            dmat = torch.cdist(local_pos, local_pos, p=2)
            inf = torch.full((n,), float("inf"), device=device)
            dmat[torch.arange(n, device=device), torch.arange(n, device=device)] = inf

            tie = (torch.arange(n, device=device, dtype=dmat.dtype) * 1e-8).view(1, -1)
            dsort = dmat + tie
            nn2 = torch.argsort(dsort, dim=1)[:, :2]

            for local_u in range(n):
                local_v = int(nn2[local_u, 0].item())
                local_w = int(nn2[local_u, 1].item())
                u = int(g_nodes[local_u].item())
                v = int(g_nodes[local_v].item())
                w = int(g_nodes[local_w].item())

                if v == u or w == u or v == w:
                    continue

                triplets.append((v, u, w))

                pv_u = self._pair_key(v, u)
                pu_w = self._pair_key(u, w)
                pv_w = self._pair_key(v, w)
                active_pairs.add(pv_u)
                active_pairs.add(pu_w)
                active_pairs.add(pv_w)

        if len(active_pairs) == 0:
            # Extremely small edge-less graph fallback: create no pairs/triplets.
            return PairTripletBatch(
                pair_indices=torch.empty((2, 0), dtype=torch.long, device=device),
                pair_dist=torch.empty((0, 1), dtype=pos.dtype, device=device),
                pair_bond_flag=torch.empty((0, 1), dtype=pos.dtype, device=device),
                pair_graph=torch.empty((0,), dtype=torch.long, device=device),
                pair_vu_idx=torch.empty((0,), dtype=torch.long, device=device),
                pair_uw_idx=torch.empty((0,), dtype=torch.long, device=device),
                pair_vw_idx=torch.empty((0,), dtype=torch.long, device=device),
                geom_features=torch.empty((0, 4), dtype=pos.dtype, device=device),
            )

        sorted_pairs = sorted(active_pairs)
        pair_to_idx: Dict[Tuple[int, int], int] = {p: i for i, p in enumerate(sorted_pairs)}

        pair_indices = torch.tensor(sorted_pairs, dtype=torch.long, device=device).t().contiguous()
        pa, pb = pair_indices[0], pair_indices[1]
        pair_dist = (pos[pa] - pos[pb]).norm(dim=-1, keepdim=True)
        pair_bond_flag = torch.tensor(
            [[1.0] if (int(a), int(b)) in bonded_pairs else [0.0] for (a, b) in sorted_pairs],
            dtype=pos.dtype,
            device=device,
        )
        pair_graph = batch[pa]

        pair_vu_idx: List[int] = []
        pair_uw_idx: List[int] = []
        pair_vw_idx: List[int] = []
        geom: List[torch.Tensor] = []

        for (v, u, w) in triplets:
            key_vu = self._pair_key(v, u)
            key_uw = self._pair_key(u, w)
            key_vw = self._pair_key(v, w)
            pair_vu_idx.append(pair_to_idx[key_vu])
            pair_uw_idx.append(pair_to_idx[key_uw])
            pair_vw_idx.append(pair_to_idx[key_vw])

            d_uv = (pos[u] - pos[v]).norm()
            d_uw = (pos[u] - pos[w]).norm()
            d_vw = (pos[v] - pos[w]).norm()
            angle = self._angle(pos[v] - pos[u], pos[w] - pos[u])
            geom.append(torch.stack([d_uv, d_uw, d_vw, angle]))

        if len(pair_vw_idx) > 0:
            pair_vu_idx_t = torch.tensor(pair_vu_idx, dtype=torch.long, device=device)
            pair_uw_idx_t = torch.tensor(pair_uw_idx, dtype=torch.long, device=device)
            pair_vw_idx_t = torch.tensor(pair_vw_idx, dtype=torch.long, device=device)
            geom_t = torch.stack(geom, dim=0)
        else:
            pair_vu_idx_t = torch.empty((0,), dtype=torch.long, device=device)
            pair_uw_idx_t = torch.empty((0,), dtype=torch.long, device=device)
            pair_vw_idx_t = torch.empty((0,), dtype=torch.long, device=device)
            geom_t = torch.empty((0, 4), dtype=pos.dtype, device=device)

        return PairTripletBatch(
            pair_indices=pair_indices,
            pair_dist=pair_dist,
            pair_bond_flag=pair_bond_flag,
            pair_graph=pair_graph,
            pair_vu_idx=pair_vu_idx_t,
            pair_uw_idx=pair_uw_idx_t,
            pair_vw_idx=pair_vw_idx_t,
            geom_features=geom_t,
        )

    def forward(self, data) -> torch.Tensor:
        x, pos, edge_index, batch = data.x, data.pos, data.edge_index, data.batch
        h_node = self.node_encoder(x)

        structs = self._build_pair_triplet_structures(pos, edge_index, batch)

        if structs.pair_indices.size(1) > 0:
            a, b = structs.pair_indices[0], structs.pair_indices[1]
            pair_input = torch.cat(
                [h_node[a], h_node[b], structs.pair_dist, structs.pair_bond_flag],
                dim=-1,
            )
            h_pair = self.pair_init(pair_input)

            for layer in self.local2fwl_layers:
                h_pair = layer(
                    h_pair,
                    structs.pair_vu_idx,
                    structs.pair_uw_idx,
                    structs.pair_vw_idx,
                    structs.geom_features,
                )

                node_msg = torch.zeros_like(h_node)
                node_msg.index_add_(0, a, h_pair)
                node_msg.index_add_(0, b, h_pair)
                h_node = h_node + self.node_fuse(torch.cat([h_node, node_msg], dim=-1))

        for conv in self.gat_layers:
            h_node = h_node + conv(h_node, edge_index)

        g = global_mean_pool(h_node, batch)
        return self.head(g)
