import torch
import torch.nn as nn
from torch import Tensor
    
from transformers import AutoModel
from typing import Optional, Tuple
        
class Encoder(nn.Module):
    """
    Args:
        model_checkpoint: str
            The model checkpoint to load the model from.
        freeze: bool
            If True, the model weights will be frozen.
        representation: str
            The type of representation to return. 
            'cls' returns the CLS token representation.
            'mean' returns the mean of all the token representations.
    """
    def __init__(self, model_checkpoint: str, freeze: bool = False,
                 representation: str = 'cls'):
        super(Encoder, self).__init__()
        self.model = AutoModel.from_pretrained(model_checkpoint)
        self.represnetation = representation
        self.freeze = freeze
        
    def get_representation(self, input_ids: Optional[Tensor], 
                           attention_mask: Optional[Tensor]) -> Tensor:
        output = None
        if input_ids is not None:
            if self.freeze:
                with torch.no_grad():
                    out = self.model(input_ids, attention_mask=attention_mask).last_hidden_state
            else:
                out = self.model(input_ids, attention_mask=attention_mask).last_hidden_state
            if self.represnetation == 'cls':
                output = out[:, 0, :]
            elif self.represnetation == 'mean':
                output = out.mean(dim=1)            
        return output
    
    def copy_weights_from_MLM_model(self, path_state_dict: str):
        state_dict_mlm = torch.load(path_state_dict, weights_only=True)
        key_state_dict_mlm = {}
        for key in state_dict_mlm:
            key_state_dict_mlm[key] = True
        state_dict_encoder = self.model.state_dict()
        for key in state_dict_encoder:
            if "roberta." + key in key_state_dict_mlm:
                state_dict_encoder[key] = state_dict_mlm["roberta." + key].clone()
        self.model.load_state_dict(state_dict_encoder)
    
    def save(self, output_dir: str):        
        state_dict = self.model.state_dict()
        state_dict = type(state_dict)(
            {k: v.clone().cpu()
             for k, v in state_dict.items()})
        self.model.save_pretrained(output_dir, state_dict=state_dict)
        
class BiEncoder(nn.Module):
    def __init__(self, model_checkpoint: str, 
                 path_weights_mlm: Optional[str] = None,
                 encoder = None, 
                 freeze: bool = False,
                 representation: str = 'cls'):
        super(BiEncoder, self).__init__()
        if encoder is None:
            self.encoder = Encoder(model_checkpoint, freeze, representation)
        else:
            self.encoder = encoder
        if path_weights_mlm is not None:
            self.encoder.copy_weights_from_MLM_model(path_weights_mlm)
        
    def forward(self, q_ids: Tensor, q_mask: Tensor, 
                doc_ids: Tensor, doc_mask: Tensor) -> Tuple[Tensor, Tensor]:
        q_rep = self.encoder.get_representation(q_ids, q_mask)
        c_rep = self.encoder.get_representation(doc_ids, doc_mask)
        return q_rep, c_rep