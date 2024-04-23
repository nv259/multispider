# Inter Deep Ensemble Model-Agnostic
import logging

import torch
from duorat.models.lm_duorat import LMDuoRATModel
from duorat.models.rat import RATLayer
from duorat.utils import registry
from duorat.types import DuoRATBatch, DuoRATEncoderBatch, RATPreprocItem
from duorat.models.utils import _flip_attention_mask

logger = logging.getLogger(__name__)


@registry.register("model", "InterDEMA")
class InterDEMAModel(LMDuoRATModel):
    """
    A LM DuoRAT model with first layer is clone multiple times
    """

    def __init__(self, preproc, encoder, decoder):
        # ascertain the number of encoder layers remain unchanged
        encoder["rat_num_layers"] = encoder["rat_num_layers"] - 1

        super().__init__(preproc=preproc, encoder=encoder, decoder=decoder)

        self.num_particles = encoder.get("num_particles", 5)

        self.list_first_rats = torch.nn.ModuleList()
        for _ in range(self.num_particles):
            particle_encoder = RATLayer(
                embed_dim=self.encoder_rat_embed_dim,
                num_heads=encoder["rat_num_heads"],
                ffn_dim=encoder["rat_ffn_dim"],
                dropout=encoder["rat_dropout"],
                attention_dropout=encoder["rat_attention_dropout"],
                relu_dropout=encoder["rat_relu_dropout"],
            )

            self.list_first_rats.append(particle_encoder)

    def compute_branch_loss(self, preproc_items: torch.List[RATPreprocItem], particle_idx: int, debug=False) -> torch.Tensor:
        duo_rat_batch = self.items_to_duo_rat_batch(preproc_items)
        decoder_batch = duo_rat_batch.decoder_batch
        
        memory, output = self.forward_branch(batch=duo_rat_batch, particle_idx=particle_idx)
        
        assert not torch.isnan(memory).any()
        assert not torch.isnan(output).any()
        
        loss = self._compute_loss(
            memory=memory,
            output=output,
            target_key_padding_mask=decoder_batch.target_key_padding_mask,
            valid_copy_mask=decoder_batch.valid_copy_mask,
            copy_target_mask=decoder_batch.copy_target_mask,
            valid_actions_mask=decoder_batch.valid_actions_mask,
            target=decoder_batch.target,
        ).mean()
        
        return loss
    
    def forward_branch(self, batch: DuoRATBatch, particle_idx: int) -> torch.Tuple[torch.Tensor]:
        source = self._encode_branch(self, batch.encoder_batch, particle_idx)
        target = self._decode(memory=source, batch=batch.decoder_batch)
        
        return source, target
        
    def _encode_branch(self, batch: DuoRATEncoderBatch, particle_idx: int) ->  torch.Tensor:
        (batch_size, _max_input_length) = batch.input_a.shape
        
        source = self.initial_encoder(
            input_a=batch.input_a,
            input_b=batch.input_b,
            input_attention_mask=batch.input_attention_mask,
            input_key_padding_mask=batch.input_key_padding_mask,
            input_token_type_ids=batch.input_token_type_ids,
            input_position_ids=batch.input_position_ids,
            input_source_gather_index=batch.input_source_gather_index,
            input_segments=batch.input_segments,
        )
        
        (_batch_size, max_src_length, _encoder_rat_embed_dim) = source.shape
        
        assert _batch_size == batch_size
        assert _encoder_rat_embed_dim == self.encoder_rat_embed_dim
        
        source = self._encode_source_branch(
            source=source,
            source_relations=batch.source_relations,
            source_attention_mask=batch.source_attention_mask,
            source_key_padding_mask=batch.source_key_padding_mask,
            particle_idx=particle_idx
        )
        assert source.shape == (batch_size, max_src_length, self.encoder_rat_embed_dim)
        return source
    
    def _encode_source_branch(
        self,
        source: torch.Tensor,
        source_relations: torch.Tensor,
        source_attention_mask: torch.Tensor,
        source_key_padding_mask: torch.Tensor,
        particle_idx: int
    ) -> torch.Tensor:
        device = next(self.parameters()).device
        source_relations = source_relations.to(device)
        (batch_size, max_src_length, _encoder_rat_embed_dim) = source.shape

        _source_relations = self.source_relation_embed(source_relations)
        assert _source_relations.shape == (
            batch_size,
            max_src_length,
            max_src_length,
            self.encoder_rat_head_dim,
        )

        _source_attention_mask = _flip_attention_mask(source_attention_mask).to(
            device=device
        )
        _source_key_padding_mask = ~source_key_padding_mask.to(device=device)

        source = self.list_first_rats[particle_idx](
            x=source,
            relations_k=_source_relations,
            relations_v=_source_relations
            if self.relation_aware_values
            else torch.zeros_like(_source_relations),
            attention_mask=_source_attention_mask,
            key_padding_mask=_source_key_padding_mask,
        )
        
        for layer in self.encoder_rat_layers:
            source = layer(
                x=source,
                relations_k=_source_relations,
                relations_v=_source_relations
                if self.relation_aware_values
                else torch.zeros_like(_source_relations),
                attention_mask=_source_attention_mask,
                key_padding_mask=_source_key_padding_mask,
            )
        assert source.shape == (batch_size, max_src_length, self.encoder_rat_embed_dim)
        return source
    
    def _encode_source(self, source: torch.Tensor, source_relations: torch.Tensor, source_attention_mask: torch.Tensor, source_key_padding_mask: torch.Tensor) -> torch.Tensor:
        device = next(self.parameters()).device
        source_relations = source_relations.to(device)
        (batch_size, max_src_length, _encoder_rat_embed_dim) = source.shape

        _source_relations = self.source_relation_embed(source_relations)
        assert _source_relations.shape == (
            batch_size,
            max_src_length,
            max_src_length,
            self.encoder_rat_head_dim,
        )

        _source_attention_mask = _flip_attention_mask(source_attention_mask).to(
            device=device
        )
        _source_key_padding_mask = ~source_key_padding_mask.to(device=device)

        source_particle_list = []
        
        for i in range(self.num_particles): 
            source = self.list_first_rats[i](
                x=source,
                relations_k=_source_relations,
                relations_v=_source_relations
                if self.relation_aware_values
                else torch.zeros_like(_source_relations),
                attention_mask=_source_attention_mask,
                key_padding_mask=_source_key_padding_mask,
            )
            
            source_particle_list.append(source)
            
        source = torch.stack(source_particle_list, dim=0).mean(dim=0)

        for layer in self.encoder_rat_layers:
            source = layer(
                x=source,
                relations_k=_source_relations,
                relations_v=_source_relations
                if self.relation_aware_values
                else torch.zeros_like(_source_relations),
                attention_mask=_source_attention_mask,
                key_padding_mask=_source_key_padding_mask,
            )
        assert source.shape == (batch_size, max_src_length, self.encoder_rat_embed_dim)
        return source