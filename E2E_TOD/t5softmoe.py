import torch
from torch import nn
from torch.nn import Parameter
from transformers.models.t5.modeling_t5 import T5Stack, T5Block, T5LayerNorm, T5LayerFF, T5DenseReluDense
from transformers.activations import  ACT2FN
from soft_mixture_of_experts.soft_moe import SoftMoE
from logging import getLogger
logger = getLogger(__name__)

class AdapterLayer(nn.Module):
    def __init__(self, dim, down_dim, norm=None):
        super().__init__()

        self.dim = dim
        self.down_dim = down_dim
        self.dropout = nn.Dropout(0.2)
        self.down = nn.Linear(dim, down_dim)
        self.relu = nn.ReLU()
        self.up = nn.Linear(down_dim, dim)
        if norm is not None:
            self.layer_norm = T5LayerNorm(dim)
            self.layer_norm.weight = Parameter(norm.clone())

    def forward(self, inputs):
        x = self.dropout(inputs)
        x = self.down(x)
        x = self.relu(x)
        x = self.up(x)
        x += inputs
        x = self.layer_norm(x)
        return x


class TaskOptimizedAdapter(nn.Module):
    def __init__(self, adapter_type, adapter_config, task: list, norm=None):
        super().__init__()
        self.toa = nn.ModuleDict({i: adapter_type(**adapter_config, norm=norm) for i in task})
        self.task = 'nlu'

    def forward(self, inputs):
        return self.toa[self.task](inputs)


class SoftMoEModuleList(nn.ModuleList):
    def __init__(self, modules):
        super().__init__()
        self += nn.ModuleList(modules)
    # freeze pretrained parameter and other task adapters of all blocks
    def freeze_pretrained(self):
        for module in self:
            module.freeze_pretrained()

class T5SoftMoEDenseReluDense(T5DenseReluDense):
    def __init__(self, dense_layer, config, num_experts, slots_per_expert):
        super().__init__(config)
        self.wi = dense_layer.wi
        self.dropout = dense_layer.dropout
        #self.wi = nn.Linear(config.d_model, config.d_ff, bias=False)
        #self.wo = nn.Linear(config.d_ff, config.d_model, bias=False)
        #self.dropout = nn.Dropout(config.dropout_rate)
        #self.act = ACT2FN[config.dense_act_fn]
        #self.wo = dense_layer.wo
        self.moe = SoftMoE(
            in_features=config.d_ff, 
            out_features=config.d_model,
            num_experts=num_experts,
            slots_per_expert=slots_per_expert,
            bias=False,
        )

    def forward(self, hidden_states):
        hidden_states = self.wi(hidden_states)
        hidden_states = nn.functional.relu(hidden_states)
        hidden_states = self.dropout(hidden_states)
        '''
        if (
            isinstance(self.moe.weight, torch.Tensor)
            and hidden_states.dtype != self.moe.weight.dtype
            and self.moe.weight.dtype != torch.int8
        ):
            hidden_states = hidden_states.to(self.moe.weight.dtype)
        '''
        hidden_states = self.moe(hidden_states)
        return hidden_states

class T5SoftMoELayerFF(T5LayerFF):
    def __init__(self, layer, config, num_experts, slots_per_expert):
        super().__init__(config)
        #if config.is_gated_act:
        #    self.DenseReluDense = T5DenseGatedActDense(config)
        #else:
        #    self.DenseReluDense = T5SoftMoEDenseActDense(config, num_experts, slots_per_expert)

        #self.layer_norm = T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        #self.dropout = nn.Dropout(config.dropout_rate)
        if config.feed_forward_proj == "relu":
            self.DenseReluDense = T5SoftMoEDenseReluDense(layer.DenseReluDense, config, num_experts, slots_per_expert)
        else:
            self.DenseReluDense = layer.DenseReluDense
        self.layer_norm = layer.layer_norm
        self.dropout = layer.dropout

    def forward(self, hidden_states):
        forwarded_states = self.layer_norm(hidden_states)
        forwarded_states = self.DenseReluDense(forwarded_states)
        hidden_states = hidden_states + self.dropout(forwarded_states)
        return hidden_states


class T5SoftMoEBlock(T5Block):
    def __init__(self, block, config, num_experts, slots_per_expert):
        super().__init__(config)
        #self.is_decoder = config.is_decoder
        #self.layer = nn.ModuleList()
        #self.layer.append(T5LayerSelfAttention(config, has_relative_attention_bias=has_relative_attention_bias))
        #if self.is_decoder:
        #    self.layer.append(T5LayerCrossAttention(config))
        self.is_decoder = block.is_decoder
        self.layer = block.layer
        self.layer[-1] = T5SoftMoELayerFF(block.layer[-1], config, num_experts, slots_per_expert)

    def forward(
            self,
            hidden_states,
            attention_mask=None,
            position_bias=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            encoder_decoder_position_bias=None,
            layer_head_mask=None,
            cross_attn_layer_head_mask=None,
            past_key_value=None,
            use_cache=False,
            output_attentions=False,
            return_dict=True,
    ):

        if past_key_value is not None:
            if not self.is_decoder:
                logger.warning("`past_key_values` is passed to the encoder. Please make sure this is intended.")
            expected_num_past_key_values = 2 if encoder_hidden_states is None else 4

            if len(past_key_value) != expected_num_past_key_values:
                raise ValueError(
                    f"There should be {expected_num_past_key_values} past states. "
                    f"{'2 (past / key) for cross attention. ' if expected_num_past_key_values == 4 else ''}"
                    f"Got {len(past_key_value)} past key / value states"
                )

            self_attn_past_key_value = past_key_value[:2]
            cross_attn_past_key_value = past_key_value[2:]
        else:
            self_attn_past_key_value, cross_attn_past_key_value = None, None

        self_attention_outputs = self.layer[0](
            hidden_states,
            attention_mask=attention_mask,
            position_bias=position_bias,
            layer_head_mask=layer_head_mask,
            past_key_value=self_attn_past_key_value,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )
        hidden_states, present_key_value_state = self_attention_outputs[:2]
        attention_outputs = self_attention_outputs[2:]  # Keep self-attention outputs and relative position weights

        # clamp inf values to enable fp16 training
        if hidden_states.dtype == torch.float16 and torch.isinf(hidden_states).any():
            clamp_value = torch.finfo(hidden_states.dtype).max - 1000
            hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

        do_cross_attention = self.is_decoder and encoder_hidden_states is not None
        if do_cross_attention:
            # the actual query length is unknown for cross attention
            # if using past key value states. Need to inject it here
            if present_key_value_state is not None:
                query_length = present_key_value_state[0].shape[2]
            else:
                query_length = None

            cross_attention_outputs = self.layer[1](
                hidden_states,
                key_value_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                position_bias=encoder_decoder_position_bias,
                layer_head_mask=cross_attn_layer_head_mask,
                past_key_value=cross_attn_past_key_value,
                query_length=query_length,
                use_cache=use_cache,
                output_attentions=output_attentions,
            )
            hidden_states = cross_attention_outputs[0]

            # clamp inf values to enable fp16 training
            if hidden_states.dtype == torch.float16 and torch.isinf(hidden_states).any():
                clamp_value = torch.finfo(hidden_states.dtype).max - 1000
                hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

            # Combine self attn and cross attn key value states
            if present_key_value_state is not None:
                present_key_value_state = present_key_value_state + cross_attention_outputs[1]

            # Keep cross-attention outputs and relative position weights
            attention_outputs = attention_outputs + cross_attention_outputs[2:]

        # Apply Feed Forward layer
        hidden_states = self.layer[-1](hidden_states)

        # clamp inf values to enable fp16 training
        if hidden_states.dtype == torch.float16 and torch.isinf(hidden_states).any():
            clamp_value = torch.finfo(hidden_states.dtype).max - 1000
            hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

        outputs = (hidden_states,)

        if use_cache:
            outputs = outputs + (present_key_value_state,) + attention_outputs
        else:
            outputs = outputs + attention_outputs

        return outputs

    # freeze pretrained params and other task adapters in the block
    def freeze_pretrained(self):
        for params in self.layer[-1].DenseReluDense.moe.parameters():
            params.requires_grad = True

    def set_layer_task(self, task):
        for layer in self.layer:
            layer.layer_norm.task = task

# add adapter to the Transformer blocks in the model
def add_adapter(model, num_experts, slots_per_expert):
    model.encoder.block = SoftMoEModuleList(
        [T5SoftMoEBlock(block, model.encoder.config, num_experts, slots_per_expert) for block in
         model.encoder.block])
    model.decoder.block = SoftMoEModuleList(
        [T5SoftMoEBlock(block, model.decoder.config, num_experts, slots_per_expert) for block in
         model.decoder.block])
    return model


# freeze pretrained parameter & other task adapters
def set_frozen_parameters_for_train(model):
    for params in model.parameters():
        params.requires_grad = True
    #model.encoder.block.freeze_pretrained()
    #model.decoder.block.freeze_pretrained()
    return model


def set_task_for_inference(model, task):
    for block in model.encoder.block:
        block.task = task
        block.set_layer_task(task)
    for block in model.decoder.block:
        block.task = task
        block.set_layer_task(task)
    return model

def copy_weight(target_model, reference_model, task):
    reference_encoder, reference_decoder = reference_model.encoder, reference_model.decoder

    for block, ref_block in zip(target_model.encoder.block,reference_encoder.block):
        for layer, ref_layer in zip(block.layer, ref_block.layer):
            for params, ref_params in zip(layer.layer_norm.toa[task].parameters(), ref_layer.layer_norm.toa[task].parameters()):
                params.data.copy_(ref_params.data)
    for block, ref_block in zip(target_model.decoder.block, reference_decoder.block):
        for layer, ref_layer in zip(block.layer, ref_block.layer):
            for params, ref_params in zip(layer.layer_norm.toa[task].parameters(), ref_layer.layer_norm.toa[task].parameters()):
                params.data.copy_(ref_params.data)

    return target_model

if __name__ == '__main__':
    from transformers import T5ForConditionalGeneration, AutoTokenizer
    model = T5ForConditionalGeneration.from_pretrained('t5-small')
    tokenizer = AutoTokenizer.from_pretrained('t5-small')
    print(model.encoder.block[0].layer[1].DenseReluDense.wi.weight)

    # add adapter to the model
    model = add_adapter(model, num_experts=32, slots_per_expert=2)
    print(model.encoder.block[0].layer[1].DenseReluDense.moe.experts.weight.shape)
