import torch
import torch.nn as nn

class PatchEmbed(nn.Module):
    """ Patch Embedding
    Splitted the image into patches and then embed them

    Parameters:
    ----------
    img_size: int
        The size of the image (assuming it is square)
    patch_size: int
        The size of the patch (assuming it is square)
    in_channels: int
        The number of input channels in the image. Like the number of color channels
    embed_dim: int
        The dimension of the embedding (output dimension)

    Attributes:
    ----------
    n_patches: int
        The number of patches in the image
    proj: nn.Conv2d
        The convolutional layer that does the both splitting into patches and the embedding

    """

    def __init__(self, img_size, patch_size, in_channels = 3, embed_dim = 768) -> None:
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2

        self.projection = nn.Conv2d(
            in_channels, embed_dim,
            kernel_size = patch_size,
            stride = patch_size
        )

    def forward(self, x):
        """
        Run the forward pass.

        Parameters:
        ----------
        x: torch.Tensor
            The input tensor. It has shape (n_samples, in_channels, img_size, img_size).

        Returns:
        -------
        torch.Tensor
            The output tensor. It has shape (n_samples, n_patches, embed_dim).
        """

        x = self.projection(x) # It will split the image into patches and then apply a linear transformation like an embedding (n_samples, embed_dim, n_patches ** 0.5, n_patches ** 0.5)
        x = x.flatten(2) # It will flatten the last two dimensions. The output tensor will have shape (n_samples, embed_dim, n_patches)
        x = x.transpose(1, 2) # It will swap the second and third dimensions. The output tensor will have shape (n_samples, n_patches, embed_dim)
        return x
    


class Attention(nn.Module):
    """ Attention Mechanism
    
    Parameters:
    ----------
    embed_dim: int
        The input and output dimension of the attention mechanism (the embedding dimension) per head (token). It is the d_model in the transformer
    n_heads: int
        The number of heads in the multi-head attention mechanism
    qkv_bias: bool
        Whether to include bias in the query, key, and value linear transformations
    attn_p: float
        The dropout probability for the attention mechanism
    proj_p: float
        The dropout probability for the output tensor

    Attributes:
    ----------
    scale: float
        The scale factor for the dot product. It is the square root of the embedding dimension per head
    qkv : nn.Linear
        The linear transformation for the query, key, and value
    proj: nn.Linear
        The linear mapping for the output tensor after the attention mechanism that combines the heads. It is the linear transformation for the output tensor. It maps the output tensor to the original embedding dimension
    attn_drop: nn.Dropout
        The dropout layer for the attention mechanism
    """

    def __init__(self, embed_dim, n_heads = 12, qkv_bias = True, attn_p = 0., proj_p = 0.) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.head_dim = embed_dim // n_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias = qkv_bias)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.attn_drop = nn.Dropout(attn_p)
        self.proj_drop = nn.Dropout(proj_p)

    def forward(self, x):
        """
        Run the forward pass.

        Parameters:
        ----------
        x: torch.Tensor
            The input tensor. It has shape (n_samples, n_patches + 1, embed_dim). The n_patches + 1 is because we have the class token

        Returns:
        -------
        torch.Tensor
            The output tensor. It has shape (n_samples, n_patches + 1, embed_dim).
        """

        n_samples, n_tokens, embed_dim = x.shape
        qkv = self.qkv(x).reshape(n_samples, n_tokens, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim = -1) # Apply the softmax to the last dimension
        attn = self.attn_drop(attn) # Apply dropout to the attention mechanism

        x = (attn @ v).transpose(1, 2).reshape(n_samples, n_tokens, embed_dim)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
    


class MLP(nn.Module):
    """ Multi-Layer Perceptron
    
    Parameters:
    ----------
    in_features: int
        The number of input features
    hidden_features: int
        The number of hidden features
    out_features: int
        The number of output features
    p: float
        The dropout probability

    Attributes:
    ----------
    fc1: nn.Linear
        The first linear transformation
    act: nn.GELU
        The GELU activation function
    fc2: nn.Linear
        The second linear transformation
    dropout: nn.Dropout
        The dropout layer
    """

    def __init__(self, in_features, hidden_features = None, out_features = None, p = 0.) -> None:
        super().__init__()
        hidden_features = hidden_features or in_features
        out_features = out_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.dropout = nn.Dropout(p)

    def forward(self, x):
        """
        Run the forward pass.

        Parameters:
        ----------
        x: torch.Tensor
            The input tensor. It has shape (n_samples, n_patches + 1, in_features). The n_patches + 1 is because we have the class token

        Returns:
        -------
        torch.Tensor
            The output tensor. It has shape (n_samples, n_patches + 1, out_features).
        """

        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x
    


class Block(nn.Module):
    """ Transformer Block
    
    Parameters:
    ----------
    embed_dim: int
        The input and output dimension of the attention mechanism (the embedding dimension) per head (token). It is the d_model in the transformer
    n_heads: int
        The number of heads in the multi-head attention mechanism
    mlp_ratio: int
        The factor to increase the hidden dimension in the feedforward block
    qkv_bias: bool
        Whether to include bias in the query, key, and value linear transformations
    attn_p: float
        The dropout probability for the attention mechanism
    proj_p: float
        The dropout probability for the output tensor
    mlp_p: float
        The dropout probability for the feedforward block

    Attributes:
    ----------
    norm1: nn.LayerNorm
        The first layer normalization layer
    attn: Attention
        The attention mechanism
    drop_path: nn.Module
        The drop path layer
    norm2: nn.LayerNorm
        The second layer normalization layer
    mlp: MLP
        The feedforward block
    """

    def __init__(self, embed_dim, n_heads, mlp_ratio = 4., qkv_bias = True, attn_p = 0., proj_p = 0., mlp_p = 0.) -> None:
        super().__init__()
        hidden_dim = int(embed_dim * mlp_ratio)
        self.norm1 = nn.LayerNorm(embed_dim, eps = 1e-6)
        self.attn = Attention(
            embed_dim,
            n_heads,
            qkv_bias,
            attn_p,
            proj_p
            )
        self.drop_path = nn.Identity()
        self.norm2 = nn.LayerNorm(embed_dim, eps=1e-6)
        hidden_features = int(embed_dim * mlp_ratio)
        self.mlp = MLP(
            in_features = embed_dim,
            hidden_features = hidden_features,
            out_features = embed_dim,
            p = mlp_p
        )

    def forward(self, x):
        """
        Run the forward pass.

        Parameters:
        ----------
        x: torch.Tensor
            The input tensor. It has shape (n_samples, n_patches + 1, embed_dim). The n_patches + 1 is because we have the class token

        Returns:
        -------
        torch.Tensor
            The output tensor. It has shape (n_samples, n_patches + 1, embed_dim).
        """

        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x
    

class VisionTransformer(nn.Module):
    """ Simplified implementation of Vision Transformer

    Parameters:
    ----------
    img_size: int
        The size of the image (assuming it is square)
    patch_size: int
        The size of the patch (assuming it is square)
    in_channels: int
        The number of input channels in the image. Like the number of color channels
    n_classes: int
        The number of classes in the dataset
    embed_dim: int
        The dimension of the embedding (output dimension)
    depth: int
        The number of transformer blocks
    n_heads: int
        The number of heads in the multi-head attention mechanism
    mlp_ratio: float
        Determines the hidden dimension in the 'MLP' block
    qkv_bias: bool
        Whether to include bias in the query, key, and value linear transformations
    attn_p: float
        The dropout probability for the attention mechanism
    proj_p: float
        The dropout probability for the output tensor

    Attributes:
    ----------
    patch_embed: PatchEmbed
        The patch embedding layer
    cls_token: nn.Parameter
        The learnable parameter that will represent the class token (first token) in the sequence. It is a tensor of shape (1, 1, embed_dim)/
    pos_embed: nn.Parameter
        The learnable parameter that will represent the positional encodding. It is a tensor of shape (1, n_patches + 1, embed_dim)
    pos_drop: nn.Dropout
        The dropout layer for the positional encodding
    blocks: nn.ModuleList
        The transformer blocks
    norm: nn.LayerNorm
        The layer normalization layer
    """

    def __init__(self, img_size = 384, patch_size = 16, in_channels = 3, n_classes = 1000, embed_dim = 768, depth = 12, n_heads = 12, mlp_ratio = 4., qkv_bias = True, attn_p = 0., proj_p = 0.):
        super().__init__()
        self.patch_embed = PatchEmbed(
            img_size = img_size,
            patch_size = patch_size,
            in_channels = in_channels,
            embed_dim = embed_dim
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.patch_embed.n_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p = proj_p)

        self.blocks = nn.ModuleList([
            Block(
                embed_dim = embed_dim,
                n_heads = n_heads,
                mlp_ratio = mlp_ratio,
                qkv_bias = qkv_bias,
                attn_p = attn_p,
                proj_p = proj_p
            ) for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(embed_dim, eps = 1e-6)
        self.head = nn.Linear(embed_dim, n_classes)

    def forward(self, x):
        """
        Run the forward pass.

        Parameters:
        ----------
        x: torch.Tensor
            The input tensor. It has shape (n_samples, in_channels, img_size, img_size).

        Returns:
        -------
        torch.Tensor
            The output tensor. It has shape (n_samples, n_classes).
        """

        n_samples = x.shape[0]
        x = self.patch_embed(x)
        cls_token = self.cls_token.expand(n_samples, -1, -1)
        x = torch.cat((cls_token, x), dim = 1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)
        x = x[:, 0] # Get the first token
        x = self.head(x)
        return x