import torch
from torch_geometric.data import Data, Batch

from qm9_local2fwl.model import Local2FWLGAT


def test_local2fwl_forward_smoke():
    x = torch.randn(4, 11)
    pos = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
    )
    edge_index = torch.tensor(
        [[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]], dtype=torch.long
    )
    y = torch.zeros((1, 19), dtype=torch.float32)

    data = Data(x=x, pos=pos, edge_index=edge_index, y=y)
    batch = Batch.from_data_list([data])

    model = Local2FWLGAT(node_in_dim=11, hidden_dim=32, pair_layers=2, gat_layers=1, gat_heads=4)
    out = model(batch)

    assert out.shape == (1, 1)
    assert torch.isfinite(out).all()
