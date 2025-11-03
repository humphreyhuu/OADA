import math
import torch
from torch import nn
from typing import Tuple


class Attention(nn.Module):
    def forward(self, query, key, value, mask=None, dropout=None):
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(query.size(-1))
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        p_attn = torch.softmax(scores, dim=-1)
        if mask is not None:
            p_attn = p_attn.masked_fill(mask == 0, 0)
        if dropout is not None:
            p_attn = dropout(p_attn)

        return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        """
        Args:
            h: number of heads
            d_model: number of dimension before multi-head
            dropout: used for dropout for attention results of each head
        """
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0

        self.d_k = d_model // h
        self.h = h

        self.linear_layers = nn.ModuleList(
            [nn.Linear(d_model, d_model, bias=False) for _ in range(3)]
        )
        self.output_linear = nn.Linear(d_model, d_model, bias=False)
        self.attention = Attention()

        self.dropout = nn.Dropout(p=dropout)

        self.attn_gradients = None
        self.attn_map = None

    # helper functions for interpretability
    def get_attn_map(self):
        return self.attn_map

    def get_attn_grad(self):
        return self.attn_gradients

    def save_attn_grad(self, attn_grad):
        self.attn_gradients = attn_grad

    # register_hook option allows us to save the gradients in backwarding
    def forward(self, query, key, value, mask=None, register_hook=False):
        batch_size = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = [
            linear(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
            for linear, x in zip(self.linear_layers, (query, key, value))
        ]

        # 2) Apply attention on all the projected vectors in batch.
        if mask is not None:
            mask = mask.unsqueeze(1)
        x, attn = self.attention(query, key, value, mask=mask, dropout=self.dropout)

        self.attn_map = attn  # save the attention map
        if register_hook:
            attn.register_hook(self.save_attn_grad)
        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)

        return self.output_linear(x)


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, x, mask=None):
        x = self.w_2(self.dropout(self.activation(self.w_1(x))))
        if mask is not None:
            mask = mask.sum(dim=-1) > 0
            x[~mask] = 0
        return x


class SublayerConnection(nn.Module):
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = nn.LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))


class TransformerBlock(nn.Module):
    """Component: MultiHeadedAttention + PositionwiseFeedForward + SublayerConnection
    Args:
        hidden: hidden size of transformer.
        attn_heads: head sizes of multi-head attention.
        dropout: dropout rate.
    """
    def __init__(self, hidden, attn_heads, dropout):
        super(TransformerBlock, self).__init__()
        self.attention = MultiHeadedAttention(h=attn_heads, d_model=hidden)
        self.feed_forward = PositionwiseFeedForward(
            d_model=hidden, d_ff=4 * hidden, dropout=dropout
        )
        self.input_sublayer = SublayerConnection(size=hidden, dropout=dropout)
        self.output_sublayer = SublayerConnection(size=hidden, dropout=dropout)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, mask=None, register_hook=False):
        """Forward propagation.
        Args:
            x: [batch_size, seq_len, hidden]
            mask: [batch_size, seq_len, seq_len]
            register_hook: whether to save the gradients of attention weights.
        Returns:
            A tensor of shape [batch_size, seq_len, hidden]
        """
        x = self.input_sublayer(x, lambda _x: self.attention(_x, _x, _x, mask=mask, register_hook=register_hook))
        x = self.output_sublayer(x, lambda _x: self.feed_forward(_x, mask=mask))
        return self.dropout(x)


class TransformerLayer(nn.Module):
    """Transformer layer.
    Paper: Ashish Vaswani et al. Attention is all you need. NIPS 2017.
    """
    def __init__(self, feature_size, heads=4, dropout=0.5, num_layers=3):
        super(TransformerLayer, self).__init__()
        self.transformer = nn.ModuleList(
            [TransformerBlock(feature_size, heads, dropout) for _ in range(num_layers)]
        )

    def forward(self, x, mask=None, register_hook=False) -> Tuple[torch.tensor, torch.tensor]:
        """Forward propagation.
        Args:
            x: a tensor of shape [batch size, sequence len, feature_size].
            mask: an optional tensor of shape [batch size, sequence len], where
                1 indicates valid and 0 indicates invalid.
            register_hook: True to save gradients of attention layer, Default is False.
        Returns:
            emb: a tensor of shape [batch size, sequence len, feature_size],
                containing the output features for each time step.
            cls_emb: a tensor of shape [batch size, feature_size], containing
                the output features for the first time step.
        """
        if mask is not None:
            mask = torch.einsum("ab,ac->abc", mask, mask)
        for transformer in self.transformer:
            x = transformer(x, mask, register_hook)
        emb = x
        cls_emb = x[:, 0, :]
        return emb, cls_emb


class Transformer(nn.Module):
    def __init__(self, feature_keys, code_nums, embedding_dim, output_size, activation=True, **kwargs):
        super(Transformer, self).__init__()
        self.feature_keys = feature_keys
        self.embedding_dim = embedding_dim
        self.code_nums = code_nums

        self.embeddings = nn.ModuleDict()
        # self.linear_layers = nn.ModuleDict()
        self.transformer = nn.ModuleDict()
        self.fc = nn.Linear(len(self.feature_keys) * self.embedding_dim, output_size)

        # Add activation layer
        self.use_activation = activation
        if self.use_activation:
            self.activation = nn.Sigmoid()

        for feature_key in feature_keys:
            self.embeddings[feature_key] = nn.Embedding(self.code_nums[feature_key]+1, embedding_dim, padding_idx=0)
            self.transformer[feature_key] = TransformerLayer(feature_size=embedding_dim, **kwargs)
            # self.linear_layers[feature_key] = nn.Linear(embedding_dim, embedding_dim)

    def forward(self, code_x, **kwargs) -> torch.Tensor:
        assert len(code_x[self.feature_keys[0]].shape) == 3
        patient_emb = []
        for feature_key in self.feature_keys:
            x = code_x[feature_key]
            x = self.embeddings[feature_key](x)  # (patient, visit, events, embedding_dim)
            x = torch.sum(x, dim=-2)  # (patient, visit, embedding_dim)
            mask = torch.any(x != 0, dim=-1)  # (patient, visit)
            _, x = self.transformer[feature_key](x, mask, kwargs.get('register_hook', False))
            patient_emb.append(x)

        patient_emb = torch.cat(patient_emb, dim=1)
        logits = self.fc(patient_emb)

        # Apply activation if specified
        if self.use_activation:
            logits = self.activation(logits)

        return logits
