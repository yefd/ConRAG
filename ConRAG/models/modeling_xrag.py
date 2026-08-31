# copy and revise from https://github.com/Hannibal046/xRAG/blob/main/src/model/xMistral/modeling_xmistral.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import re
from transformers import AutoModelForCausalLM, PretrainedConfig, PreTrainedModel
from typing import Optional,Union


def get_kl_loss(teacher_logits, student_logits, teacher_labels, student_labels, temperature=1.0, distill_topk=None):
    
    ## make sure the teacher_logits and student_logits have the same shape
    loss_fct = nn.KLDivLoss(reduction="batchmean")
    _,_,vocab_size = student_logits.shape

    ## only compute loss in the completion part, not propmt
    student_mask = (student_labels!=-100).unsqueeze(-1).expand_as(student_logits) ## batch_size,num_tokens,vocab_size
    student_logits_selected = torch.masked_select(student_logits,student_mask).view(-1,vocab_size)

    teacher_mask = (teacher_labels != -100).unsqueeze(-1).expand_as(teacher_logits)
    teacher_logits_selected = torch.masked_select(teacher_logits,teacher_mask).view(-1,vocab_size)

    if distill_topk is not None:
        _, topk_teacher_indices = torch.topk(teacher_logits_selected, k=distill_topk, dim=-1)  
      
        teacher_logits_selected = torch.gather(teacher_logits_selected, 1, topk_teacher_indices)  
        student_logits_selected = torch.gather(student_logits_selected, 1, topk_teacher_indices) 

    assert teacher_logits_selected.shape == student_logits_selected.shape, (f"The shape of teacher logits is {teacher_logits_selected.shape}, while that of student is {student_logits_selected.shape}")

    kl_loss = loss_fct(
        F.log_softmax(student_logits_selected / temperature, dim=-1),
        F.softmax(    teacher_logits_selected / temperature, dim=-1),
    ) * temperature ** 2
    
    return kl_loss

class XRAGConfig(PretrainedConfig):
    def __init__(
        self,
        model_name_or_path='',
        projector_type='mlp2x_gelu',
        input_dim=768,
        hidden_size=4096,
        use_flash_att=False,
        xrag_token='<unk>',
        xrag_token_id=0,
        freeze_llm=True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.model_name_or_path = model_name_or_path
        self.projector_type = projector_type
        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.xrag_token = xrag_token
        self.xrag_token_id = xrag_token_id
        self.use_flash_att = use_flash_att
        self.freeze_llm = freeze_llm


class Projector(nn.Module):
    def __init__(self,config):
        super().__init__()
        projector_type = config.projector_type
        mlp_gelu_match = re.match(r'^mlp(\d+)x_gelu$', projector_type)
        if mlp_gelu_match:
            mlp_depth = int(mlp_gelu_match.group(1))
            modules = [nn.Linear(config.input_dim, config.hidden_size)]
            for _ in range(1, mlp_depth):
                modules.append(nn.GELU())
                modules.append(nn.Linear(config.hidden_size, config.hidden_size))
            self.projector = nn.Sequential(*modules)
    
    def forward(self,context_embedding):
        return self.projector(context_embedding)

class XRAGForCausalLM(PreTrainedModel):
    config_class = XRAGConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    def __init__(self,config):
        super().__init__(config)
        self.xrag_token_id = config.xrag_token_id
        self.alpha_kl = 2
        if config.use_flash_att:
            self.llama_model = AutoModelForCausalLM.from_pretrained(
                config.model_name_or_path, 
                device_map="auto",
                attn_implementation="flash_attention_2",
                torch_dtype=torch.bfloat16,
                )
        else:
            self.llama_model = AutoModelForCausalLM.from_pretrained(
                config.model_name_or_path, 
                device_map="auto",
                )
        print('freeze_llm')
        for name, param in self.llama_model.named_parameters():
            param.requires_grad = False
        if hasattr(config,"input_dim") and config.input_dim > 0: 
            self.projector = Projector(config).to(self.llama_model.device)
            self.input_dim = config.input_dim
        # self.post_init()

    def get_input_embeddings(self):
        return self.llama_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.llama_model.set_input_embeddings(value)

    def get_output_embeddings(self):
        return self.llama_model.get_output_embeddings()

    def set_output_embeddings(self, new_embeddings):
        self.llama_model.set_output_embeddings(new_embeddings)

    def set_decoder(self, decoder):
        self.llama_model.set_decoder(decoder)

    def get_decoder(self):
        return self.llama_model.get_decoder()
    
    def set_xrag_token_id(self,token_id):
        self.xrag_token_id = token_id

    def prepare_inputs_embeds(self, input_ids, embeds):
        embed_tokens = self.get_input_embeddings()
        inputs_embeds = embed_tokens(input_ids)
        # embeds = embeds.view(-1, self.input_dim)
        if embeds.dim() <= 2 and self.config.input_dim == 1:
            embeds = embeds.unsqueeze(-1)

        ## sanity check
        # num_xrag_tokens = torch.sum(input_ids==self.xrag_token_id).item()
        # num_retrieval_embeds = retrieval_embeds.shape[0]
        # assert num_xrag_tokens == num_retrieval_embeds, (num_xrag_tokens, num_retrieval_embeds)
        
        embeds = self.projector(embeds.to(inputs_embeds.dtype))
        inputs_embeds[input_ids==self.xrag_token_id] = embeds.to(inputs_embeds.dtype)
        return inputs_embeds

    def forward(
        self,
        input_ids = None,
        embeds = None, ## [-1,retrieval_hidden_size]
        attention_mask = None,
        **kwargs,
    ):
        # inputs_embeds = kwargs.pop("inputs_embeds", None)
        # at_the_beginning_of_generation = False
        # if inputs_embeds is not None:
        #     assert not self.training
        #     assert embeds is None
        #     at_the_beginning_of_generation = True

        # if not at_the_beginning_of_generation:
        #     ## a single forward
        #     if embeds is not None:
        #         inputs_embeds = self.prepare_inputs_embeds(input_ids, embeds)
        #         input_ids = None
        #         if attention_mask is not None:
        #             assert inputs_embeds.shape[1] == attention_mask.shape[1],(inputs_embeds.shape,attention_mask.shape)

        # return super().forward(
        #     input_ids = input_ids,
        #     inputs_embeds = inputs_embeds,
        #     attention_mask = attention_mask,
        #     **kwargs,
        # )
        inputs_embeds = self.prepare_inputs_embeds(input_ids, embeds)
        # print(kwargs.keys())
        if 'inputs_embeds' in kwargs:
            _ = kwargs.pop('inputs_embeds')
        if 'label' in kwargs:
            teacher_input_ids = kwargs.pop('label')
        outputs = self.llama_model(inputs_embeds=inputs_embeds, **kwargs)
        if teacher_input_ids.dim() > 1 and teacher_input_ids.shape[1] > 1:
            labels = kwargs['labels']
            teacher_labels = torch.full((teacher_input_ids.shape[0], teacher_input_ids.shape[1]), -100).to(labels.device)
            teacher_labels[:, -labels.shape[1]:] = labels
            self.llama_model.eval()
            teacher_outputs = self.llama_model(input_ids=teacher_input_ids)
            self.llama_model.train()
            kl_loss = get_kl_loss(teacher_outputs.logits, outputs.logits, teacher_labels, labels) * self.alpha_kl
            outputs.loss += kl_loss
            # raise ValueError
        return outputs

    @torch.no_grad()
    def generate(
        self,
        inputs: torch.Tensor,
        input_ids = None,
        embeds = None,
        **kwargs,
    ):
        # attention_mask = kwargs.pop("attention_mask", None)
        # if "inputs_embeds" in kwargs:
        #     raise NotImplementedError("`inputs_embeds` is not supported for generate")
        
        # inputs_embeds=None
        # if embeds is not None:
        #     inputs_embeds = self.prepare_inputs_embeds(input_ids,embeds)
        #     input_ids = None
        #     if attention_mask is not None:
        #         assert inputs_embeds.shape[1] == attention_mask.shape[1],(inputs_embeds.shape,attention_mask.shape)
        #     return super().generate(
        #         attention_mask=attention_mask,
        #         inputs_embeds=inputs_embeds,
        #         **kwargs
        #     )
        
        # else:
        #     return super().generate(
        #         attention_mask=attention_mask,
        #         input_ids=input_ids,
        #         **kwargs
        #     )
        if 'embeds' not in inputs.keys():
            raise ValueError('embeds is None')
        if 'label' in kwargs:
            _ = kwargs.pop('label')
        if 'inputs_embeds' in kwargs:
            _ = kwargs.pop('inputs_embeds')
        # print('input_ids', input_ids.shape)
        inputs_embeds = self.prepare_inputs_embeds(inputs['input_ids'], inputs['embeds'])
        outputs = self.llama_model.generate(inputs_embeds=inputs_embeds, **kwargs)
        return outputs
    