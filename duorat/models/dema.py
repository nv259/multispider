# Inter Deep Ensemble Model-Agnostic
import logging

import torch

from duorat.models.rat import RATLayer
from duorat.models.lm_duorat import LMDuoRATModel
from duorat.utils import registry

logger = logging.getLogger(__name__)


@registry.register("model", "InterDEMA")
class InterDEMA(LMDuoRATModel):
    """
    A LM DuoRAT model with first layer is clone multiple times 
    """ 
    def __init__(self, preprocess, encoder, decoder, num_particles=5):
        
        # ascertain the number of encoder remain unchanged
        encoder["rat_num_layers"] = encoder["rat_num_layer"] - 1 
        
        super().__init__(
            preproc=preprocess,
            encoder=encoder,
            decoder=decoder
        )
        
        self.num_particles = num_particles
         
        self.list_first_rats = torch.nn.ModuleList()
        for _ in range(num_particles):
            particle_encoder = RATLayer(
                embed_dim=self.decoder_rat_embed_dim,
                mem_embed_dim=self.mem_embed_dim,
                num_heads=decoder["rat_num_heads"],
                ffn_dim=decoder["rat_ffn_dim"],
                dropout=decoder["rat_dropout"],
                attention_dropout=decoder["rat_attention_dropout"],
                relu_dropout=decoder["rat_relu_dropout"]
            )
            
            self.list_first_rats.append(particle_encoder)
        
        