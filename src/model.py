from typing import Dict, Mapping, Optional,Any
import torch
import numpy as np
from torch import nn, Tensor
import torch.nn.functional as F

from tqdm import trange
from mamba_ssm.modules.mamba_simple import Mamba
from mamba_ssm.modules.mamba2 import Mamba2



class BiMambaWrapper(nn.Module):
    """Thin wrapper around Mamba to support bi-directionality."""
    
    def __init__( 
            self,
            d_model: int,
            bidirectional: bool = True,
            bidirectional_strategy: str = "add",
            bidirectional_weight_tie: bool = True, 
            ssm_layer: str = "Mamba2",
            **mamba_kwargs
    ):
        super().__init__()
        
        # Validate bidirectional strategy
        if bidirectional and bidirectional_strategy not in ["add", "ew_multiply"]:
            raise NotImplementedError(f"`{bidirectional_strategy}` strategy for bi-directionality is not implemented!")
            
        self.bidirectional = bidirectional
        self.bidirectional_strategy = bidirectional_strategy
        
        # Choose SSM layer
        self.mamba = Mamba2 if ssm_layer == "Mamba2" else Mamba
        
        # Initialize forward Mamba
        self.mamba_fwd = self.mamba(
            d_model=d_model,
            **mamba_kwargs
        )
        
        # Initialize reverse Mamba if bidirectional
        if bidirectional:
            self.mamba_rev = self.mamba(
                d_model=d_model,
                **mamba_kwargs
            )
            
            # Weight tying (optional)
            if bidirectional_weight_tie:
                self.mamba_rev.in_proj.weight = self.mamba_fwd.in_proj.weight
                self.mamba_rev.in_proj.bias = self.mamba_fwd.in_proj.bias
                self.mamba_rev.out_proj.weight = self.mamba_fwd.out_proj.weight
                self.mamba_rev.out_proj.bias = self.mamba_fwd.out_proj.bias
        else:
            self.mamba_rev = None  
    def forward(self, hidden_states, src_key_padding_mask=None, inference_params=None):
        """Bidirectional-enabled forward pass with smart flipping
        
        Args:
            hidden_states: (B, L, D)
            src_key_padding_mask: (B, L), optional. True for padding positions.
        Returns:
            same shape as hidden_states
        """
        # Forward pass
        hidden_states = F.layer_norm(hidden_states, hidden_states.shape[-1:])
        out = self.mamba_fwd(hidden_states, inference_params=inference_params)
        
        # Bidirectional processing
        if self.bidirectional:
            # Create default mask if not provided (all False, meaning no padding)
            if src_key_padding_mask is None:
                src_key_padding_mask = torch.zeros_like(hidden_states[:, :, 0], dtype=torch.bool)
            
            # Smart reverse sequence dimension (handles padding properly)
            reversed_states = self.smart_flip(hidden_states, src_key_padding_mask)
            
            # Reverse Mamba pass
            out_rev = self.mamba_rev(
                reversed_states,
                inference_params=inference_params
            )
            # Flip back using smart flip (using the same mask)
            out_rev = self.smart_flip(out_rev, src_key_padding_mask)
            
            # Combine strategies
            if self.bidirectional_strategy == "add":
                out = (out + out_rev)/2
            elif self.bidirectional_strategy == "ew_multiply":
                out = out * out_rev
            else:
                raise NotImplementedError(f"`{self.bidirectional_strategy}` strategy not implemented!")
        
        out = F.layer_norm(out, out.shape[-1:])
        return out

    def smart_flip(self, X, src_key_padding_mask):
        batch_size, seq_length, embedding_dim = X.size()
        # Calculate the actual lengths of sequences without padding
        lengths = (~src_key_padding_mask).sum(dim=1)  # Note the ~ to invert the mask
        # Create a range tensor that will be used for indexing
        range_tensor = torch.arange(seq_length, device=X.device).unsqueeze(0).expand(batch_size, -1)
        # Create the flip mask
        flip_mask = range_tensor < lengths.unsqueeze(1)
        # Reverse the positions within each sequence
        reversed_positions = torch.where(flip_mask, lengths.unsqueeze(1) - 1 - range_tensor, range_tensor)
        # Gather the tensor according to reversed positions
        X_flipped = torch.gather(X, 1, reversed_positions.unsqueeze(-1).expand(-1, -1, embedding_dim))
        return X_flipped



class MambaModel(nn.Module):
    def __init__(
        self,
        ntoken: int,
        d_model: int,
        nhead: int,         
        d_hid: int,           
        nlayers: int,         
        vocab: Any = None,
        dropout: float = 0.1,
        pad_token: str = "<pad>",
        pad_value: int = 0,
        input_emb_style: str = "continuous",
        n_input_bins: Optional[int] = None,
        cell_emb_style: str = "cls",
        bidirectional: bool = True,
        bidirectional_strategy: str = "add",
        bidirectional_weight_tie: bool = True,
        class_num: int=164
    ):
        super().__init__()
        self.model_type = "Mamba2"
        self.d_model = d_model
        self.input_emb_style = input_emb_style
        self.cell_emb_style = cell_emb_style
        self.class_num=class_num
        self.encoder = GeneEncoder(ntoken, d_model, padding_idx=60694)
        

        self.value_encoder = ContinuousValueEncoder(d_model, dropout)
        self.cls_decoder = ClsDecoder(d_model, n_cls=class_num,nlayers=3)
        self.decoder = ExprDecoder(d_model)
        self.creterion_cce = nn.CrossEntropyLoss()
        self.mamba_encoder = nn.ModuleList(
            [BiMambaWrapper(
                d_model=d_model,
                bidirectional=bidirectional,
                bidirectional_strategy=bidirectional_strategy,
                bidirectional_weight_tie=bidirectional_weight_tie,
                ssm_layer="Mamba2",
            ) for _ in range(nlayers)]
        )
        

    def _encode(self, src: Tensor, values: Tensor, src_key_padding_mask: Tensor) -> Tensor:
        src = self.encoder(src)
        values = self.value_encoder(values)
        total_embs = src + values  
        
        if src_key_padding_mask is not None:
            total_embs = total_embs * (~src_key_padding_mask).unsqueeze(-1)
        
    
        for layer in self.mamba_encoder:
            total_embs = layer(total_embs, src_key_padding_mask=src_key_padding_mask)
        
        return total_embs
    def _get_cell_emb_from_layer(self, layer_output: Tensor, weights: Tensor = None) -> Tensor:
        if self.cell_emb_style == "cls":
            cell_emb = layer_output[:, 0, :]
        elif self.cell_emb_style == "avg-pool":
            cell_emb = torch.mean(layer_output, dim=1)
        elif self.cell_emb_style == "w-pool":
            if weights is None:
                raise ValueError("weights is required when cell_emb_style is w-pool")
            cell_emb = torch.sum(layer_output * weights.unsqueeze(2), dim=1)
            cell_emb = F.normalize(cell_emb, p=2, dim=1)
        elif self.cell_emb_style == "all-genes":
            cell_emb = layer_output[:, 1:, :] 
        return cell_emb

    def forward(self, src: Tensor, values: Tensor, src_key_padding_mask: Tensor) -> Mapping[str, Tensor]:
        transformer_output = self._encode(src, values, src_key_padding_mask)
        output = {}
        mlm_output = self.decoder(transformer_output)
        output["mlm_output"] = mlm_output["pred"]
        cell_emb = self._get_cell_emb_from_layer(transformer_output, values)
        output["cell_emb"] = cell_emb
        cls_output = self.cls_decoder(cell_emb)
        output["cls_output"] = cls_output
        return output


class GeneEncoder(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: Optional[int] = None,
    ):
        super().__init__()
        self.embedding = nn.Embedding(
            num_embeddings, embedding_dim, padding_idx=padding_idx
        )
        self.enc_norm = nn.LayerNorm(embedding_dim)

    def forward(self, x: Tensor) -> Tensor:
        x = self.embedding(x)  # (batch, seq_len, embsize)
        x = self.enc_norm(x)
        return x

class ContinuousValueEncoder(nn.Module):
    """
    Encode real number values to a vector using neural nets projection.
    """

    def __init__(self, d_model: int, dropout: float = 0.1, max_value: int = 512):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.linear1 = nn.Linear(1, d_model)
        self.activation = nn.ReLU()
        self.linear2 = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.max_value = max_value

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, seq_len]
        """
        x = x.unsqueeze(-1)
        x = torch.clamp(x, max=self.max_value)
        x = self.activation(self.linear1(x))
        x = self.linear2(x)
        x = self.norm(x)
        return self.dropout(x)


class ExprDecoder(nn.Module):
    def __init__(
        self,
        d_model: int,
    ):
        super().__init__()
        d_in = d_model
        self.fc = nn.Sequential(
            nn.Linear(d_in, d_model),
            nn.LeakyReLU(),
            nn.Linear(d_model, d_model),
            nn.LeakyReLU(),
            nn.Linear(d_model, 1),
        )

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        """x is the output of the transformer, (batch, seq_len, d_model)"""
        pred_value = self.fc(x).squeeze(-1)  # (batch, seq_len)
        return dict(pred=pred_value)
       
class ClsDecoder(nn.Module):
    """
    Decoder for classification task.
    """

    def __init__(
        self,
        d_model: int,
        n_cls: int,
        nlayers: int = 3,
        activation: callable = nn.ReLU,
    ):
        super().__init__()
        # module list
        self._decoder = nn.ModuleList()
        for i in range(nlayers - 1):
            self._decoder.append(nn.Linear(d_model, d_model))
            self._decoder.append(activation())
            self._decoder.append(nn.LayerNorm(d_model))
        self.out_layer = nn.Linear(d_model, n_cls)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, embsize]
        """
        for layer in self._decoder:
            x = layer(x)
        return self.out_layer(x)
