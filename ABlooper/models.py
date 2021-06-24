import torch
from torch import nn, einsum, broadcast_tensors
from einops import rearrange

# Most of the code in this file is based on egnn-pytorch by lucidrains.

class Swish_(nn.Module):
    def forward(self, x):
        return x * x.sigmoid()


SiLU = nn.SiLU if hasattr(nn, 'SiLU') else Swish_


class CoorsNorm(nn.Module):
    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps
        self.fn = nn.LayerNorm(1)

    def forward(self, coors):
        norm = coors.norm(dim=-1, keepdim=True)
        normed_coors = coors / norm.clamp(min=self.eps)
        phase = self.fn(norm)
        return phase * normed_coors


# classes

class EGNN(nn.Module):
    def __init__(
            self,
            dim,
            m_dim=32,
    ):
        super().__init__()

        edge_input_dim = (dim * 2) + 1

        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_input_dim, edge_input_dim * 2),
            SiLU(),
            nn.Linear(edge_input_dim * 2, m_dim),
            SiLU()
        )

        self.coors_norm = CoorsNorm()

        self.node_mlp = nn.Sequential(
            nn.Linear(dim + m_dim, dim * 2),
            SiLU(),
            nn.Linear(dim * 2, dim),
        )

        self.coors_mlp = nn.Sequential(
            nn.Linear(m_dim, m_dim * 4),
            SiLU(),
            nn.Linear(m_dim * 4, 1)
        )

    def forward(self, feats, coors):
        rel_coors = rearrange(coors, 'b i d -> b i () d') - rearrange(coors, 'b j d -> b () j d')
        rel_dist = (rel_coors ** 2).sum(dim=-1, keepdim=True)

        feats_j = rearrange(feats, 'b j d -> b () j d')
        feats_i = rearrange(feats, 'b i d -> b i () d')
        feats_i, feats_j = broadcast_tensors(feats_i, feats_j)

        edge_input = torch.cat((feats_i, feats_j, rel_dist), dim=-1)

        m_ij = self.edge_mlp(edge_input)

        coor_weights = self.coors_mlp(m_ij)
        coor_weights = rearrange(coor_weights, 'b i j () -> b i j')

        rel_coors = self.coors_norm(rel_coors)

        coors_out = einsum('b i j, b i j c -> b i c', coor_weights, rel_coors) + coors

        m_i = m_ij.sum(dim=-2)

        node_mlp_input = torch.cat((feats, m_i), dim=-1)
        node_out = self.node_mlp(node_mlp_input) + feats

        return node_out, coors_out


class ResEGNN(nn.Module):
    def __init__(self, corrections=4, dims_in=41, **kwargs):
        super().__init__()
        self.layers = nn.ModuleList([EGNN(dim=dims_in, **kwargs) for _ in range(corrections)])

    def forward(self, amino, geom):
        for layer in self.layers:
            amino, geom = layer(amino, geom)
        return geom


class DecoyGen(nn.Module):
    def __init__(self, dims_in=41, decoys=5, **kwargs):
        super().__init__()
        self.blocks = nn.ModuleList([ResEGNN(dims_in=dims_in, **kwargs) for _ in range(decoys)])
        self.decoys = decoys

    def forward(self, amino, geom):
        geoms = torch.zeros((self.decoys, *geom.shape[1:]), device=geom.device)

        for i, block in enumerate(self.blocks):
            geoms[i] = block(amino, geom)

        return geoms
