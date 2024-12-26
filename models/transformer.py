import torch
import torch.nn as nn
import torch.nn.functional as F

from .fcn import MLP

class MultiHeadAttention(nn.Module):
    """
    Multiple Attention Heads. Now support causal attention.
    
    Args:
        input_dim: The dimension of input tokens.
        num_heads: The number of heads.
        head_dim: The dimension of each head.
        out_dim: The dimension of output tokens.
        
    """
    def __init__(
        self, input_dim, num_heads, head_dim, out_dim
    ):
        super().__init__()

        self.input_dim = input_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.inner_dim = head_dim * num_heads

        self.key=nn.Parameter(
            torch.randn( self.inner_dim, self.input_dim)
        )
        self.query=nn.Parameter(
            torch.randn( self.inner_dim, self.input_dim)
        )
        self.value=nn.Parameter(
            torch.randn( self.inner_dim, self.input_dim)
        )
        self.projection=nn.Parameter(
            torch.randn( self.out_dim, self.inner_dim)
        )

    def forward(self, x, causal_mask=False):
        """
        Args:
            x: input, tensor of size (batch_size, input_size, input_dim).
        
        Returns:
            The output of a multi-head attention layer,
            of size (batch_size, input_size, output_dim)
        """
        B,T,C = x.size()
        k = F.linear( x, self.key, bias=None).view(B, T, self.num_heads, self.head_dim).transpose(1,2) *C**-.5    # [bs, num_heads, seq_len, head_dim]
        q = F.linear( x, self.query, bias=None).view(B, T, self.num_heads, self.head_dim).transpose(1,2) *C**-.5  # [bs, num_heads, seq_len, head_dim]
        v = F.linear( x, self.value, bias=None).view(B, T, self.num_heads, self.head_dim).transpose(1,2) *C**-.5  # [bs, num_heads, seq_len, head_dim]

        weight = q @ k.transpose(-2,-1) * self.head_dim**-.5             # [bs, num_heads, seq_len, seq_len]

        if causal_mask:
            causal_mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
            weight = weight.masked_fill(causal_mask, float('-inf'))

        
        weight = F.softmax(weight, dim=-1)                              #  //

        out = (weight @ v).transpose(1,2).reshape(B,T,-1) # [bs, seq_len, inner_dim]
        out = F.linear( out, self.projection, bias=None) * self.projection.size(-1)**-.5

        return out


class AttentionBlock(nn.Module):
    def __init__(
        self, embedding_dim, num_heads
    ):
        super().__init__()
        assert embedding_dim % num_heads == 0, "embedding dim. must be multiple of num. heads"

        self.sa = MultiHeadAttention(
            input_dim=embedding_dim,
            num_heads=num_heads,
            head_dim=embedding_dim//num_heads,
            out_dim=embedding_dim
        )

    def forward(self, x):
        x = self.sa(x, causal_mask=False)
        return x
    
class CausalAttentionBlock(nn.Module):
    def __init__(
        self, embedding_dim, num_heads
    ):
        super().__init__()
        assert embedding_dim % num_heads == 0, "embedding dim. must be multiple of num. heads"

        self.sa = MultiHeadAttention(
            input_dim=embedding_dim,
            num_heads=num_heads,
            head_dim=embedding_dim//num_heads,
            out_dim=embedding_dim
        )

    def forward(self, x):
        x = self.sa(x, causal_mask=True)
        return x


class EncoderBlock(nn.Module):
    """
    One Decoder Block.
    
    Args:
        embedding_dim: The dimension of the tokens (kept constant past embedding).
        num_heads: The number of attention heads.
        ffwd_size: Size of the MLP is ffwd_size*embedding_dim.
    """
    def __init__(
        self, embedding_dim, num_heads, ffwd_size=4
    ):
        super().__init__()
        assert embedding_dim % num_heads == 0, "embedding dim. must be multiple of num. heads"

        self.sa = MultiHeadAttention(
            input_dim=embedding_dim,
            num_heads=num_heads,
            head_dim=embedding_dim//num_heads,
            out_dim=embedding_dim
        )
        self.ffwd = MLP(
            input_dim=embedding_dim, 
            nn_dim=ffwd_size*embedding_dim, 
            out_dim=embedding_dim, 
            num_layers=1
        )
        self.ln1 = nn.LayerNorm(embedding_dim)
        self.ln2 = nn.LayerNorm(embedding_dim)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class MLA(nn.Module):
    """
    Multi-Layer Multi-Head Attention for last token prediction

    Args:
        vocab_size: The dimension of input tokens.
        block_size: The (maximal) number of input tokens.
        embedding_dim: The embedding dimension.
        num_heads: The number of attention heads.
        num_layers: The number of layers.
        causal: Whether or not to use causal attention. (default=False)
        keep_time: Whether or not to keep the time axis of the input. (default=False)

    Returns:
        model(x).shape = (batch, vocab_dim) if keep_time=False or (batch, vocab_dim, seq_len)
        if keep_time=True (note: this is a permutation of the input shape!)

    """
    def __init__(
        self, vocab_size, block_size, embedding_dim, num_heads, num_layers, causal=False, keep_time=False
    ):
        super().__init__()

        self.attention_block = CausalAttentionBlock if causal else AttentionBlock

        self.vocab_size = vocab_size
        self.block_size = block_size
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        
        self.keep_time=keep_time

        self.token_embedding=nn.Parameter(
            torch.randn( self.embedding_dim, self.vocab_size)
        )
        self.position_embedding = nn.Embedding(self.block_size, self.embedding_dim)

        self.blocks = nn.Sequential(
            *[
                self.attention_block(
                    embedding_dim=self.embedding_dim,
                    num_heads=self.num_heads,
                ) for _ in range(self.num_layers)
            ]
        )
        self.readout = nn.Parameter(
            torch.randn(self.vocab_size, self.embedding_dim)
        )


    def forward(self, x):
        """
        Args:
            x: input, tensor of size (batch_size, seq_len, vocab_size).
        
        Returns:
            Output of multilayer self-attention, tensor of size (batch_size, seq_len, vocab_size)
        """
        B,T,C = x.size()
        token_emb = F.linear( x, self.token_embedding, bias=None) *C**-.5 # [bs, seq_len, embedding_dim]
        pos_emb = self.position_embedding(torch.arange(T, device=x.device)) # [seq_len, embedding_dim]
        x = token_emb + pos_emb  # [bs, seq_len, embedding_dim]
        x = self.blocks(x)
        if self.keep_time:
            logits = ( F.linear(x, self.readout, bias=None) * self.readout.size(-1)**-.5 ).permute(0,2,1) # [bs, vocab_dim, seq_len]
        else:
            logits = F.linear( x[:,-1,:], self.readout, bias=None) * self.readout.size(-1)**-.5 # [bs, vocab_dim]

        return logits
