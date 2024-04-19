import attr
import math
import torch
import torch.nn as nn
import numpy as np
import torch.autograd as autograd 
import operator
import collections
import gc 
import copy
import logging

logger = logging.getLogger("tensor2struct")

# Todo: move this class for universal use
@attr.s
class SpiderEncoderState:
    state = attr.ib()
    memory = attr.ib()
    question_memory = attr.ib()
    schema_memory = attr.ib()
    words_for_copying = attr.ib()

    pointer_memories = attr.ib()
    pointer_maps = attr.ib()

    m2c_align_mat = attr.ib()
    m2t_align_mat = attr.ib()

    # for copying
    tokenizer = attr.ib()

    def find_word_occurrences(self, token):
        occurrences = [i for i, w in enumerate(self.words_for_copying) if w == token]
        if len(occurrences) > 0:
            return occurrences[0]
        else:
            return None

class InterDeepEnsembleModelAgnostic(nn.Module):
    def __init__(
        self,  
        device=None, 
        num_particles=2,
    ):
        super().__init__()
        self.inner_steps = 1
        self.num_particles = num_particles
        self.device = device

    def ensemble_train(self, 
                       model, 
                       batch, 
                       num_batch_accumulated):
        
        return self.particles_base_train(model, 
                                         batch,
                                         num_batch_accumulated)

    def particles_base_train(self, 
                             model, 
                             batch, 
                             num_batch_accumulated,
                             prior_scale=1e-3):
        assert model.training
        
        inner_inter_params = []
        for i in range(self.num_particles):
            inner_inter_params.append(list(model.list_first_rats[i].parameters()))
            
        params_matrix = torch.stack(
            [torch.nn.utils.parameters_to_vector(params) for params in inner_inter_params],
            dim=0
        )
        bert_len = len(list(model.bert_model.parameters()))
        particle_len = len(list(model.list_first_rats[0].parameters()))
        encoder_len = len(list(model.encoder.parameters()))
        aligner_len = len(list(model.aligner.parameters()))
        ret_dic = {}
        # for computing distance 
        distance_nll = torch.empty(size=(self.num_particles,
                                            params_matrix.size(1)),
                                    device=self.device)
        enc_input_list = [enc_input for enc_input, dec_output in batch]
        column_pointer_maps = [
            {i: [i] for i in range(len(desc["columns"]))} for desc in enc_input_list
        ]
        table_pointer_maps = [
            {i: [i] for i in range(len(desc["tables"]))} for desc in enc_input_list
        ]

        final_losses = []
        bert_grads = None
        encoder_grads = None
        aligner_grads = None
        decoder_grads = None            
            
        for i in range(self.num_particles):
            
            plm_output = model.bert_model(enc_input_list)
            # for single input source domain
            enc_states = []
            for idx, (enc_input, plm_out) in enumerate(zip(enc_input_list, plm_output)):
                relation = model.schema_linking(enc_input)
                (
                    enc_new, 
                    c_base,
                    t_base,
                ) = model.list_first_rats[i](enc_input, plm_out, relation)
                
                (
                    q_enc_new_item,
                    c_enc_new_item,
                    t_enc_new_item,
                ) = model.encoder(enc_new, relation, c_base, t_base)
                # attention memory 
                memory = []
                include_in_memory = model.encoder.include_in_memory
                if "question" in include_in_memory:
                    memory.append(q_enc_new_item)
                if "column" in include_in_memory:
                    memory.append(c_enc_new_item)
                if "table" in include_in_memory:
                    memory.append(t_enc_new_item)
                memory = torch.cat(memory, dim=1)
                # alignment matrix
                align_mat_item = model.aligner(
                    enc_input, q_enc_new_item, c_enc_new_item, t_enc_new_item, relation
                )
                enc_states.append(
                    SpiderEncoderState(
                        state=None,
                        words_for_copying=enc_input["question_for_copying"],
                        tokenizer=model.encoder.tokenizer,
                        memory=memory,
                        question_memory=q_enc_new_item,
                        schema_memory=torch.cat((c_enc_new_item, t_enc_new_item), dim=1),
                        pointer_memories={
                            "column": c_enc_new_item,
                            "table": t_enc_new_item,
                        },
                        pointer_maps={
                            "column": column_pointer_maps[idx],
                            "table": table_pointer_maps[idx],
                        },
                        m2c_align_mat=align_mat_item[0],
                        m2t_align_mat=align_mat_item[1],
                    )
                )

            losses = []
            for enc_state, (enc_input, dec_output) in zip(enc_states, batch):
                ret_dic = model.decoder(dec_output, enc_state)
                losses.append(ret_dic["loss"])
            loss = torch.mean(torch.stack(losses, dim=0), dim=0) / num_batch_accumulated
            final_losses.append(loss.item()*num_batch_accumulated)
            grads = torch.autograd.grad(loss, 
                                        list(model.bert_model.parameters())
                                        + list(model.list_first_rats[i].parameters()) 
                                        + list(model.encoder.parameters()) 
                                        + list(model.aligner.parameters())
                                        + list(model.decoder.parameters()),
                                        allow_unused=True)
            
            particle_grads = grads[bert_len:bert_len+particle_len]
            if aligner_grads is None:
                bert_grads = grads[:bert_len]
                encoder_grads = grads[bert_len
                                      +particle_len:bert_len
                                      +particle_len
                                      +encoder_len]
                aligner_grads = grads[bert_len
                                      +particle_len
                                      +encoder_len:
                                      bert_len
                                      +particle_len
                                      +encoder_len
                                      +aligner_len]
                decoder_grads = grads[bert_len
                                      +particle_len
                                      +encoder_len
                                      +aligner_len:]
            else:
                bert_grads = tuple(x+y if y is not None else None 
                                 for x,y in zip(bert_grads, grads[:bert_len])) 
                encoder_grads = tuple(x+y if y is not None else None 
                                 for x,y in zip(encoder_grads, grads[bert_len
                                      +particle_len:bert_len
                                      +particle_len
                                      +encoder_len])) 
                
                aligner_grads = tuple(x+y if y is not None else None 
                                 for x,y in zip(aligner_grads,
                                                grads[bert_len
                                      +particle_len
                                      +encoder_len:
                                      bert_len
                                      +particle_len
                                      +encoder_len
                                      +aligner_len]))
                decoder_grads = tuple(x+y if y is not None else None 
                                 for x,y in zip(decoder_grads, 
                                                grads[bert_len
                                      +particle_len
                                      +encoder_len
                                      +aligner_len:])) 

            distance_nll[i, :] = torch.nn.utils.parameters_to_vector(particle_grads)
        
        kernel_matrix, grad_kernel, _ = InterDeepEnsembleModelAgnostic.get_kernel(params=params_matrix,
                                            num_of_particles=self.num_particles)
        
        # compute inner gradients with rbf kernel
        # SVGD + prior_scale*params_matrix
        encoders_grads = (1/self.num_particles)*(torch.matmul(kernel_matrix, distance_nll) - grad_kernel)
        # wSGLD_B
        # encoders_grads = distance_nll - grad_kernel
        # copy inner_grads to main network
        for i in range(self.num_particles):
            for p_tar, p_src in zip(model.list_first_rats[i].parameters(),
                                    InterDeepEnsembleModelAgnostic.vector_to_list_params(encoders_grads[i],
                                                                                            model.list_first_rats[i].parameters())):
                p_tar.grad.data.add_(p_src) # todo: divide by num_of_sample if inner is in ba
        # # copy bert grads
        for p_tar, p_src in zip(model.bert_model.parameters(),
                                          bert_grads):
            if p_src is not None:
                p_tar.grad.data.add_(1/self.num_particles*p_src)
            else:
                p_tar.grad.data.add_(torch.zeros_like(p_tar))
        # copy encoder grads 
        for p_tar, p_src in zip(model.encoder.parameters(),
                                encoder_grads):
            if p_src is not None:
                p_tar.grad.data.add_(1/self.num_particles*p_src)
            else:
                p_tar.grad.data.add_(torch.zeros_like(p_tar))
        # copy aligner grads
        for p_tar, p_src in zip(model.aligner.parameters(),
                                aligner_grads):
            if p_src is not None:
                p_tar.grad.data.add_(1/self.num_particles*p_src)
            else:
                p_tar.grad.data.add_(torch.zeros_like(p_tar))
        # copy decoder grads
        for p_tar, p_src in zip(model.decoder.parameters(),
                                decoder_grads):
            if p_src is not None:
                p_tar.grad.data.add_(1/self.num_particles*p_src)
            else:
                p_tar.grad.data.add_(torch.zeros_like(p_tar))
        # trying to free gpu memory 
        # not sure it would help
        # del kernel_matrix
        # del grad_kernel
        # del distance_nll
        # del alinger_grads_vec
        # gc.collect()
        # torch.cuda.empty_cache()
            
        ret_dic["loss"] = sum(final_losses)/self.num_particles
        
        return ret_dic
    
    @staticmethod
    def get_kernel(params: torch.Tensor, num_of_particles):
        """
        Compute the RBF kernel for the input
        
        Args:
            params: a tensor of shape (N, M)
        
        Returns: kernel_matrix = tensor of shape (N, N)
        """
        pairwise_d_matrix = InterDeepEnsembleModelAgnostic.get_pairwise_distance_matrix(x=params)

        median_dist = torch.quantile(input=pairwise_d_matrix, q=0.5)  # tf.reduce_mean(euclidean_dists) ** 2
        h = median_dist / np.log(num_of_particles)

        kernel_matrix = torch.exp(-pairwise_d_matrix / h)
        kernel_sum = torch.sum(input=kernel_matrix, dim=1, keepdim=True)
        grad_kernel = -torch.matmul(kernel_matrix, params)
        grad_kernel += params * kernel_sum
        grad_kernel /= h

        return kernel_matrix, grad_kernel, h

    @staticmethod
    def get_kernel_wSGLD_B(params: torch.Tensor, num_of_particles):
        """
        Compute the RBF kernel and repulsive term for wSGLD 
        
        Args:
            params: a tensor of shape (N, M)
        
        Returns: 
            - kernel_matrix = tensor of shape (N, N)
            - repulsive term 
        """
        pairwise_d_matrix = InterDeepEnsembleModelAgnostic.get_pairwise_distance_matrix(x=params)
        median_dist = torch.quantile(input=pairwise_d_matrix, q=0.5)
        h = median_dist / np.log(num_of_particles)
        kernel_matrix = torch.exp(-pairwise_d_matrix / h)
        kernel_sum = torch.sum(input=kernel_matrix, dim=1, keepdim=True)
        # compute repulsive term of w_SGLD_B 
        # invert of kernel_sum Nx1 
        invert_kernel_sum = kernel_sum.pow_(-1) 
        grad_kernel = params*(torch.matmul(kernel_matrix, invert_kernel_sum) +
                                torch.sum(kernel_matrix*invert_kernel_sum, dim=1, keepdim=True))
        grad_kernel += -(torch.matmul(kernel_matrix*torch.transpose(invert_kernel_sum,0,1), params) +
                         torch.matmul(kernel_matrix, params)*invert_kernel_sum)
        grad_kernel /= h
        
        return kernel_matrix, grad_kernel, h
        
        
    @staticmethod
    def get_pairwise_distance_matrix(x: torch.Tensor) -> torch.Tensor:
        """Calculate the pairwise distance between each row of tensor x
        
        Args:
            x: input tensor
        
        Return: matrix of point-wise distances
        """
        n, m = x.shape

        # initialize matrix of pairwise distances as a N x N matrix
        pairwise_d_matrix = torch.zeros(size=(n, n), device=x.device)

        # num_particles = particle_tensor.shape[0]
        euclidean_dists = torch.nn.functional.pdist(input=x, p=2) # shape of (N)

        # assign upper-triangle part
        triu_indices = torch.triu_indices(row=n, col=n, offset=1)
        pairwise_d_matrix[triu_indices[0], triu_indices[1]] = euclidean_dists

        # assign lower-triangle part
        pairwise_d_matrix = torch.transpose(pairwise_d_matrix, dim0=0, dim1=1)
        pairwise_d_matrix[triu_indices[0], triu_indices[1]] = euclidean_dists

        return pairwise_d_matrix
    
    @staticmethod
    def vector_to_list_params(vector, other_params):
    
        params = []
        
        # pointer for each layer params
        pointer = 0 
        
        for param in other_params:
            # total number of params each layer
            num_params = int(np.prod(param.shape))
            
            params.append(vector[pointer:pointer+num_params].view(param.shape))
            
            pointer += num_params

        return params
    
IDEMA = InterDeepEnsembleModelAgnostic