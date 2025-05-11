from functools import partial
from typing import Dict, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
# from .recurrent import RNN, SimpleCell
from flax.nnx.nn.recurrent import RNN, SimpleCell
from jax import lax
from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_flax_outputs import (FlaxBaseModelOutput,
                                                FlaxTokenClassifierOutput)
from transformers.modeling_flax_utils import ACT2FN
from transformers.modeling_rope_utils import rope_config_validation

from tady.model.attention import get_attention, get_attention_lite
from tady.model.disasm_jax import DisasmJax


class TAGNNConfig(PretrainedConfig):

    model_type = "tagnn"

    def __init__(
        self,
        vocab_size: int = 2048,
        hidden_size: int = 128,
        intermediate_size: int = 256,
        num_hidden_layers: int = 4,
        num_attention_heads: int = 4,
        num_key_value_heads=None,
        connections: Dict[str, Tuple[int, int]] = {
            "must_transfer": (0, 1),
            "may_transfer": (1, 3),
            "next": (3, 4),
            "opterlapping": (4, 18)
        },
        num_global_connections: int = 32,
        global_connection_class: bool = True,
        hidden_act="silu",
        max_position_embeddings: int = 128,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        sliding_window: Tuple[int, int] = (-1, -1),
        rope_theta=10000.0,
        rope_scaling=None,
        attention_bias=False,
        attention_dropout=0.0,
        attention_type="graph",
        mlp_bias=False,
        token_pool="rnn",
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads

        # for backward compatibility
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads

        self.num_key_value_heads = num_key_value_heads
        self.connections = connections
        self.global_connection_class = global_connection_class
        # self.num_global_connections = sum(
        #     [b - a for a, b in connections.values()]) if num_global_connections != -1 else -1
        self.num_global_connections = sum(
            [b - a for a, b in connections.values()]) if connections else -1
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.attention_type = attention_type
        self.mlp_bias = mlp_bias

        self.sliding_window = sliding_window

        self.token_pool = token_pool

        # Validate the correctness of rotary position embeddings parameters
        # BC: if there is a 'type' field, move it to 'rope_type'.
        if self.rope_scaling is not None and "type" in self.rope_scaling:
            self.rope_scaling["rope_type"] = self.rope_scaling["type"]
        rope_config_validation(self)
        super().__init__(**kwargs)

def create_sinusoidal_positions(num_pos, dim, dtype=np.float32):
    inv_freq = 1.0 / (10000 ** (np.arange(0, dim, 2, dtype=np.int32) / dim))
    freqs = np.einsum("i , j -> i j", np.arange(num_pos, dtype=np.int32),
                      inv_freq).astype(dtype)

    emb = np.concatenate((freqs, freqs), axis=-1)
    out = np.concatenate(
        (np.sin(emb)[:, None, :], np.cos(emb)[:, None, :]), axis=-1)
    return jnp.array(out[:, :, :num_pos])


def rotate_half(tensor):
    """Rotates half the hidden dims of the input."""
    rotate_half_tensor = jnp.concatenate(
        (-tensor[..., tensor.shape[-1] // 2:], tensor[..., : tensor.shape[-1] // 2]), axis=-1
    )
    return rotate_half_tensor


def apply_rotary_pos_emb(tensor, sin_pos, cos_pos):
    return (tensor * cos_pos) + (rotate_half(tensor) * sin_pos)


class FlaxLlamaRMSNorm(nnx.Module):

    def __init__(self, config: TAGNNConfig,
                 dtype: jnp.dtype = jnp.float32, *, rngs: nnx.Rngs):
        self.config = config
        self.dtype = dtype
        self.epsilon = self.config.rms_norm_eps
        self.weight = nnx.Param(jnp.ones(self.config.hidden_size, dtype=dtype))

    def __call__(self, hidden_states):
        variance = jnp.asarray(hidden_states, dtype=jnp.float32)
        variance = jnp.power(variance, 2)
        variance = variance.mean(-1, keepdims=True)
        # use `jax.numpy.sqrt` as `jax.lax.rsqrt` does not match `torch.rsqrt`
        hidden_states = hidden_states / jnp.sqrt(variance + self.epsilon)

        return self.weight * jnp.asarray(hidden_states, dtype=self.dtype)


class FlaxLlamaRotaryEmbedding(nnx.Module):
    def __init__(self, config: TAGNNConfig,
                 dtype: jnp.dtype = jnp.float32, *, rngs: nnx.Rngs):
        self.config = config
        self.dtype = dtype
        head_dim = self.config.hidden_size // self.config.num_attention_heads
        self.sincos = nnx.Variable(create_sinusoidal_positions(
            self.config.max_position_embeddings, head_dim, dtype=self.dtype))

    def __call__(self, key, query, position_ids):
        sincos = self.sincos[position_ids]
        sin_pos, cos_pos = jnp.split(sincos, 2, axis=-1)
        key = apply_rotary_pos_emb(key, sin_pos, cos_pos)
        query = apply_rotary_pos_emb(query, sin_pos, cos_pos)

        key = jnp.asarray(key, dtype=self.dtype)
        query = jnp.asarray(query, dtype=self.dtype)

        return key, query


class FlaxLlamaMLP(nnx.Module):

    def __init__(self, config: TAGNNConfig,
                 dtype: jnp.dtype = jnp.float32, *, rngs: nnx.Rngs):
        self.config = config
        self.dtype = dtype
        embed_dim = self.config.hidden_size
        inner_dim = self.config.intermediate_size if self.config.intermediate_size is not None else 4 * embed_dim

        kernel_init = jax.nn.initializers.normal(
            self.config.initializer_range, dtype=self.dtype)
        self.act = ACT2FN[self.config.hidden_act]

        self.gate_proj = nnx.Linear(embed_dim,
                                    inner_dim, use_bias=False, dtype=self.dtype, param_dtype=self.dtype, kernel_init=kernel_init, rngs=rngs)
        self.down_proj = nnx.Linear(inner_dim,
                                    embed_dim, use_bias=False, dtype=self.dtype, param_dtype=self.dtype, kernel_init=kernel_init, rngs=rngs)
        self.up_proj = nnx.Linear(embed_dim, inner_dim, use_bias=False,
                                  dtype=self.dtype, param_dtype=self.dtype, kernel_init=kernel_init, rngs=rngs)

    def __call__(self, hidden_states):
        up_proj_states = self.up_proj(hidden_states)
        gate_states = self.act(self.gate_proj(hidden_states))

        hidden_states = self.down_proj(up_proj_states * gate_states)
        return hidden_states


class FlaxLlamaAttention(nnx.Module):

    def __init__(self, config: TAGNNConfig,
                 dtype: jnp.dtype = jnp.float32,
                 causal: bool = True,
                 is_cross_attention: bool = False, *, rngs: nnx.Rngs):
        self.config = config
        self.dtype = dtype
        self.causal = causal
        self.is_cross_attention = is_cross_attention
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.attention_softmax_in_fp32 = self.dtype is not jnp.float32

        dense = partial(
            nnx.Linear,
            in_features=self.config.hidden_size,
            use_bias=config.attention_bias,
            dtype=self.dtype,
            param_dtype=self.dtype,
            kernel_init=jax.nn.initializers.normal(
                self.config.initializer_range, dtype=self.dtype),
            rngs=rngs
        )

        self.q_proj = dense(out_features=self.num_heads * self.head_dim)
        self.k_proj = dense(
            out_features=self.num_key_value_heads * self.head_dim)
        self.v_proj = dense(
            out_features=self.num_key_value_heads * self.head_dim)
        self.o_proj = dense(out_features=self.embed_dim)
        self.rotary_emb = FlaxLlamaRotaryEmbedding(
            config, dtype=self.dtype, rngs=rngs)

    def _split_heads(self, hidden_states, num_heads):
        return hidden_states.reshape(hidden_states.shape[:2] + (num_heads, self.head_dim))

    def _merge_heads(self, hidden_states):
        return hidden_states.reshape(hidden_states.shape[:2] + (self.embed_dim,))

    def __call__(
        self,
        hidden_states,
        attention_mask,
        position_ids,
        rngs: Optional[nnx.Rngs] = None,
        deterministic: bool = True,
        output_attentions: bool = False,
    ):
        query = self.q_proj(hidden_states)
        key = self.k_proj(hidden_states)
        value = self.v_proj(hidden_states)

        query = self._split_heads(query, self.num_heads)
        key = self._split_heads(key, self.num_key_value_heads)
        value = self._split_heads(value, self.num_key_value_heads)
        key, query = self.rotary_emb(key, query, position_ids)

        dropout_rng = None
        if not deterministic and self.config.attention_dropout > 0.0:
            dropout_rng = rngs.dropout()

        key = jnp.repeat(key, self.num_key_value_groups, axis=2)
        value = jnp.repeat(value, self.num_key_value_groups, axis=2)
        key, value = map(lambda t: jnp.pad(
            t, ((0, 0), self.config.sliding_window, (0, 0), (0, 0))), (key, value))
        idx = jnp.arange(sum(self.config.sliding_window) + 1, dtype=jnp.int32)[
            None, :] + jnp.arange(hidden_states.shape[1], dtype=jnp.int32)[:, None]
        key, value = map(lambda t: t[:, idx, :, :], (key, value))

        batch_size, sequence_lengths = hidden_states.shape[:2]
        attention_mask = jnp.broadcast_to(
            attention_mask[:, :, None, :, :],
            (batch_size, sequence_lengths, self.config.num_attention_heads, 1, key.shape[2]))
        # usual dot product attention
        attention_dtype = jnp.float32 if self.attention_softmax_in_fp32 else self.dtype
        # transform boolean mask into float mask
        attention_bias = lax.select(
            attention_mask > 0,
            jnp.full(attention_mask.shape, 0.0).astype(self.dtype),
            jnp.full(attention_mask.shape, jnp.finfo(
                self.dtype).min).astype(self.dtype),
        )
        attn_output = nnx.dot_product_attention(
            jnp.expand_dims(query, 2), key, value,
            bias=attention_bias,
            dropout_rng=dropout_rng,
            dropout_rate=self.config.attention_dropout,
            deterministic=deterministic,
            dtype=attention_dtype).squeeze(2)
        # attn_output = jnp.einsum("...hqk,...khd->...qhd", attn_weights, value)
        attn_output = self._merge_heads(attn_output)
        attn_output = self.o_proj(attn_output)

        # outputs = (attn_output, attn_weights) if output_attentions else (
        #     attn_output,)
        outputs = (attn_output,)
        return outputs


class FlaxSelectiveAttention(nnx.Module):

    def __init__(self, config: TAGNNConfig,
                 dtype: jnp.dtype = jnp.float32,
                 causal: bool = True,
                 is_cross_attention: bool = False, *, rngs: nnx.Rngs):
        self.config = config
        self.dtype = dtype
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_global_connections = config.num_global_connections
        self.head_dim = self.embed_dim // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.attention_softmax_in_fp32 = self.dtype is not jnp.float32

        self.q_proj = nnx.Linear(
            in_features=self.config.hidden_size,
            out_features=self.num_heads * self.head_dim,
            use_bias=config.attention_bias,
            dtype=self.dtype,
            param_dtype=self.dtype,
            kernel_init=jax.nn.initializers.normal(
                self.config.initializer_range, dtype=self.dtype),
            rngs=rngs
        )
        if self.config.global_connection_class:
            self.k_projs = nnx.LinearGeneral(
                in_features=(self.num_global_connections,
                             self.config.hidden_size),
                out_features=(self.num_global_connections,
                              self.num_key_value_heads * self.head_dim),
                axis=(-2, -1),
                use_bias=config.attention_bias,
                dtype=self.dtype,
                param_dtype=self.dtype,
                kernel_init=jax.nn.initializers.normal(
                    self.config.initializer_range, dtype=self.dtype),
                rngs=rngs
            )
            self.v_projs = nnx.LinearGeneral(
                in_features=(self.num_global_connections,
                             self.config.hidden_size),
                out_features=(self.num_global_connections,
                              self.num_key_value_heads * self.head_dim),
                axis=(-2, -1),
                use_bias=config.attention_bias,
                dtype=self.dtype,
                param_dtype=self.dtype,
                kernel_init=jax.nn.initializers.normal(
                    self.config.initializer_range, dtype=self.dtype),
                rngs=rngs
            )
        else:
            self.k_projs = nnx.Linear(
                in_features=self.config.hidden_size,
                out_features=self.num_key_value_heads * self.head_dim,
                use_bias=config.attention_bias,
                dtype=self.dtype,
                param_dtype=self.dtype,
                kernel_init=jax.nn.initializers.normal(
                    self.config.initializer_range, dtype=self.dtype),
                rngs=rngs
            )
            self.v_projs = nnx.Linear(
                in_features=self.config.hidden_size,
                out_features=self.num_key_value_heads * self.head_dim,
                use_bias=config.attention_bias,
                dtype=self.dtype,
                param_dtype=self.dtype,
                kernel_init=jax.nn.initializers.normal(
                    self.config.initializer_range, dtype=self.dtype),
                rngs=rngs
            )

        self.o_proj = nnx.Linear(
            in_features=self.config.hidden_size,
            out_features=self.embed_dim,
            use_bias=config.attention_bias,
            dtype=self.dtype,
            param_dtype=self.dtype,
            kernel_init=jax.nn.initializers.normal(
                self.config.initializer_range, dtype=self.dtype),
            rngs=rngs
        )
        self.rotary_emb = FlaxLlamaRotaryEmbedding(
            config, dtype=self.dtype, rngs=rngs)

    def _split_heads(self, hidden_states, num_heads):
        return hidden_states.reshape(hidden_states.shape[:3] + (num_heads, self.head_dim))

    def _merge_heads(self, hidden_states):
        return hidden_states.reshape(hidden_states.shape[:2] + (self.embed_dim,))

    def __call__(
        self,
        hidden_states,
        connections,
        rngs: nnx.Rngs = None,
        deterministic: bool = True,
    ):
        query = self.q_proj(hidden_states)
        connections = jnp.concatenate(
            [connections[..., a:b] for a, b in self.config.connections.values()], axis=-1)
        connected_hidden_states = jnp.where(
            connections[..., None] == -1,
            0,
            jnp.take_along_axis(
                hidden_states[:, :, None, :], connections[..., None], axis=1)
        )
        key = self.k_projs(connected_hidden_states)
        value = self.v_projs(connected_hidden_states)
        # key = jnp.stack([proj(connected_hidden_states[:, :, i, :]) for proj, i in zip(
        #     self.k_projs, range(self.num_global_connections))], axis=2)
        # value = jnp.stack([proj(connected_hidden_states[:, :, i, :]) for proj, i in zip(
        #     self.v_projs, range(self.num_global_connections))], axis=2)
        query = self._split_heads(jnp.expand_dims(query, 2), self.num_heads)
        key = self._split_heads(key, self.num_key_value_heads)
        value = self._split_heads(value, self.num_key_value_heads)
        # key, query = self.rotary_emb(key, query, position_ids)

        query_length, key_length = query.shape[1], key.shape[1]

        dropout_rng = None
        if not deterministic and self.config.attention_dropout > 0.0:
            dropout_rng = rngs.dropout()

        key = jnp.repeat(key, self.num_key_value_groups, axis=-2)
        value = jnp.repeat(value, self.num_key_value_groups, axis=-2)

        # usual dot product attention
        attention_dtype = jnp.float32 if self.attention_softmax_in_fp32 else self.dtype

        attn_output = nnx.dot_product_attention(
            query,
            key,
            value,
            dropout_rng=dropout_rng,
            dropout_rate=self.config.attention_dropout,
            deterministic=deterministic,
            dtype=attention_dtype,
        ).squeeze(2)
        # attn_output = jnp.einsum("...hqk,...khd->...qhd", attn_weights, value)
        attn_output = self._merge_heads(attn_output)
        attn_output = self.o_proj(attn_output)
        outputs = (attn_output,)
        return outputs


class FlaxLlamaEncoderLayer(nnx.Module):

    def __init__(self, config: TAGNNConfig,
                 dtype: jnp.dtype = jnp.float32, *, rngs: nnx.Rngs):
        self.config = config
        self.dtype = dtype
        self.input_layernorm = FlaxLlamaRMSNorm(
            self.config, dtype=self.dtype, rngs=rngs)
        self.self_attn = FlaxLlamaAttention(
            self.config, dtype=self.dtype, rngs=rngs)
        if self.config.num_global_connections != -1:
            self.selective_attn = FlaxSelectiveAttention(
                self.config, dtype=self.dtype, rngs=rngs)
        self.post_attention_layernorm = FlaxLlamaRMSNorm(
            self.config, dtype=self.dtype, rngs=rngs)
        self.mlp = FlaxLlamaMLP(self.config, dtype=self.dtype, rngs=rngs)

    def __call__(
        self,
        hidden_states,
        attention_mask=None,
        position_ids=None,
        connections=None,
        rngs: nnx.Rngs = None,
        deterministic: bool = True,
        output_attentions: bool = False,
    ):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        outputs = self.self_attn(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            rngs=rngs,
            deterministic=deterministic,
            output_attentions=output_attentions,
        )
        # residual connection
        attn_output = outputs[0]
        if self.config.num_global_connections != -1:
            outputs = self.selective_attn(
                hidden_states,
                connections,
                rngs=rngs,
                deterministic=deterministic,
            )
            selective_output = outputs[0]
            residual = residual + selective_output
        residual = residual + attn_output

        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        # residual connection
        hidden_states = residual + hidden_states

        return (hidden_states,) + outputs[1:]


class FlaxLlamaLayerCollection(nnx.Module):

    def __init__(self, config: TAGNNConfig,
                 dtype: jnp.dtype = jnp.float32, *, rngs: nnx.Rngs):
        self.config = config
        self.dtype = dtype
        self.blocks = [
            FlaxLlamaEncoderLayer(self.config, dtype=self.dtype, rngs=rngs)
            for i in range(self.config.num_hidden_layers)
        ]

    def __call__(
        self,
        hidden_states,
        attention_mask=None,
        position_ids=None,
        connections=None,
        rngs: nnx.Rngs = None,
        deterministic: bool = True,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
    ):
        all_attentions = () if output_attentions else None
        all_hidden_states = () if output_hidden_states else None

        for block in self.blocks:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            layer_outputs = block(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                connections=connections,
                rngs=rngs,
                deterministic=deterministic,
                output_attentions=output_attentions,
            )
            hidden_states = layer_outputs[0]

            if output_attentions:
                all_attentions += (layer_outputs[1],)

        # this contains possible `None` values - `FlaxLlamaModule` will filter them out
        outputs = (hidden_states, all_hidden_states, all_attentions)

        return outputs


class FlaxLlamaModule(nnx.Module):
    def __init__(self, config: TAGNNConfig,
                 dtype: jnp.dtype = jnp.float32, *, rngs: nnx.Rngs):
        self.config = config
        self.dtype = dtype
        self.hidden_size = self.config.hidden_size
        embedding_init = jax.nn.initializers.normal(
            stddev=self.config.initializer_range)
        self.embed_tokens = nnx.Embed(
            self.config.vocab_size,
            self.hidden_size,
            embedding_init=embedding_init,
            dtype=self.dtype,
            param_dtype=self.dtype,
            rngs=rngs,
        )
        if self.config.token_pool == "rnn":
            cell = SimpleCell(
                in_features=config.hidden_size,
                hidden_features=config.hidden_size,
                activation_fn=ACT2FN[config.hidden_act],
                dtype=self.dtype,
                param_dtype=self.dtype,
                rngs=rngs
            )
            self.initial_carry = nnx.Variable(cell.initialize_carry(
                input_shape=(self.hidden_size,), rngs=rngs))
            self.pool_tokens = RNN(
                cell=cell,
                return_carry=True,
            )

        self.layers = FlaxLlamaLayerCollection(
            self.config, dtype=self.dtype, rngs=rngs)
        self.norm = FlaxLlamaRMSNorm(self.config, dtype=self.dtype, rngs=rngs)

    def __call__(
        self,
        input_ids,
        attention_mask=None,
        position_ids=None,
        connections=None,
        token_mask=None,
        rngs: nnx.Rngs = None,
        deterministic=True,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ):
        input_embeds = self.embed_tokens(input_ids.astype("i4"))
        if token_mask is None:
            token_mask = jnp.ones_like(input_ids)
        if self.config.token_pool == "rnn":
            seq_lengths = jnp.sum(token_mask, axis=-1, keepdims=False)
            input_embeds = self.pool_tokens(
                jnp.reshape(input_embeds, (-1, ) + input_embeds.shape[2:]),
                initial_carry=jnp.repeat(
                    self.initial_carry[None, :], input_embeds.shape[0] * input_embeds.shape[1], axis=0),
                seq_lengths=jnp.reshape(seq_lengths, (-1, )),
                # rngs=rngs
            )[0].reshape(input_embeds.shape[:2] + (self.config.hidden_size,))
        elif self.config.token_pool == "mean":
            if token_mask is not None:
                input_embeds = jnp.sum(
                    input_embeds, axis=2, where=jnp.expand_dims(token_mask, -1))

                # Compute valid token counts by summing attention_mask
                valid_token_counts = jnp.sum(
                    token_mask, axis=2, keepdims=True) + 1e-9  # Avoid division by zero

                # Compute average of valid embeddings
                input_embeds = input_embeds / valid_token_counts
            else:
                input_embeds = jnp.mean(input_embeds, axis=2)
        elif self.config.token_pool == "sum":
            input_embeds = jnp.sum(
                input_embeds, axis=2)

        outputs = self.layers(
            input_embeds,
            position_ids=position_ids,
            attention_mask=attention_mask,
            connections=connections,
            rngs=rngs,
            deterministic=deterministic,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        hidden_states = outputs[0]
        hidden_states = self.norm(hidden_states)

        if output_hidden_states:
            all_hidden_states = outputs[1] + (hidden_states,)
            outputs = (hidden_states, all_hidden_states) + outputs[2:]
        else:
            outputs = (hidden_states,) + outputs[1:]

        if not return_dict:
            return tuple(v for v in outputs if v is not None)

        return FlaxBaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=outputs[1],
            attentions=outputs[-1],
        )


class FlaxLlamaForTokenClassification(nnx.Module):
    def __init__(self, config: TAGNNConfig,
                 dtype: jnp.dtype = jnp.float32, *, rngs: nnx.Rngs):
        self.config = config
        self.dtype = dtype
        self.hidden_size = self.config.hidden_size
        self.model = FlaxLlamaModule(config, dtype=dtype, rngs=rngs)
        if getattr(config, "classifier_dropout", None) is not None:
            classifier_dropout = config.classifier_dropout
        else:
            classifier_dropout = 0.1
        self.dropout = nnx.Dropout(classifier_dropout)
        self.score = nnx.Linear(
            config.hidden_size, config.num_labels, dtype=self.dtype, param_dtype=self.dtype, rngs=rngs)

    def __call__(
        self,
        input_ids,
        attention_mask=None,
        connections=None,
        token_mask=None,
        rngs: nnx.Rngs = None,
        deterministic=True,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ):
        batch_size, seq_len = input_ids.shape[:2]
        position_ids = jnp.broadcast_to(jnp.arange(seq_len, dtype=jnp.int32)[
            None, :], (batch_size, seq_len))
        outputs = self.model(
            input_ids,
            token_mask=token_mask,
            attention_mask=attention_mask,
            position_ids=position_ids,
            connections=connections,
            rngs=rngs,
            deterministic=deterministic,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )
        if not return_dict:
            sequence_output = outputs[0]
        else:
            sequence_output = outputs.last_hidden_state
        sequence_output = self.dropout(
            sequence_output, deterministic=deterministic, rngs=rngs)
        logits = self.score(sequence_output)

        if not return_dict:
            outputs = (logits,) + outputs[1:]
            return tuple(v for v in outputs if v is not None)

        return FlaxTokenClassifierOutput(
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class FlaxLlamaForBinaryTokenClassification(nnx.Module):
    def __init__(self, config: TAGNNConfig,
                 dtype: jnp.dtype = jnp.float32, *, rngs: nnx.Rngs):
        self.config = config
        self.dtype = dtype
        self.hidden_size = self.config.hidden_size
        self.disassembler = DisasmJax()
        self.model = FlaxLlamaModule(config, dtype=dtype, rngs=rngs)
        if getattr(config, "classifier_dropout", None) is not None:
            classifier_dropout = config.classifier_dropout
        else:
            classifier_dropout = 0.1
        self.dropout = nnx.Dropout(classifier_dropout)
        self.score = nnx.Linear(
            config.hidden_size, config.num_labels, dtype=self.dtype, param_dtype=self.dtype, rngs=rngs)

    def __call__(
        self,
        byte_sequence: jnp.ndarray,
        use_64_bit: jnp.ndarray = None,
        rngs: nnx.Rngs = None,
        deterministic=True,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ):
        if use_64_bit is None:
            use_64_bit = jnp.ones(
                byte_sequence.shape[0], dtype=jnp.bool_)
        batch_size, seq_len = byte_sequence.shape[:2]
        input_ids, connections, instr_len, _ = nnx.vmap(
            self.disassembler, in_axes=(0, 0))(byte_sequence, use_64_bit)
        idx = jnp.arange(input_ids.shape[-1], dtype=jnp.int32)[None, None, :]
        token_mask = jnp.where(
            instr_len[:, :, None] > idx, True, False).astype(jnp.bool_)
        
        if self.config.attention_type == "graph":
            attention_mask = get_attention(
                connections[..., :4], self.config.sliding_window
            )
        elif self.config.attention_type == "lite":
            attention_mask = jax.vmap(get_attention_lite, in_axes=(0, None))(connections[..., :4], self.config.sliding_window)
        elif self.config.attention_type == "full":
            attention_mask = jnp.ones(
                byte_sequence.shape[:2] + (1, sum(self.config.sliding_window) + 1), dtype=jnp.bool)
        position_ids = jnp.broadcast_to(jnp.arange(seq_len, dtype=jnp.int32)[
            None, :], (batch_size, seq_len))
        outputs = self.model(
            input_ids,
            token_mask=token_mask,
            attention_mask=attention_mask,
            position_ids=position_ids,
            connections=connections,
            rngs=rngs,
            deterministic=deterministic,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )
        if not return_dict:
            sequence_output = outputs[0]
        else:
            sequence_output = outputs.last_hidden_state
        sequence_output = self.dropout(
            sequence_output, deterministic=deterministic, rngs=rngs)
        logits = self.score(sequence_output)

        if not return_dict:
            outputs = (logits,) + outputs[1:]
            return tuple(v for v in outputs if v is not None)

        return FlaxTokenClassifierOutput(
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

def main():
    """
    Main function to demonstrate the TAGNN model with random inputs.
    """
    num_attention_heads = 4
    hidden_size = 32
    intermediate_size = 32
    window_size = 32
    batch_size = 32
    max_position_embeddings = 8192
    node_size = 40
    num_neighbors = 1
    num_conflicts = 14
    num_embeddings = 256 * 6
    num_hidden_layers = 4
    rngs = nnx.Rngs(params=0)
    config = TAGNNConfig(
        num_attention_heads=num_attention_heads,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        window_size=window_size,
        max_position_embeddings=max_position_embeddings,
        vocab_size=num_embeddings,
        num_hidden_layers=num_hidden_layers,
        attention_type="graph",
        token_pool="mean",
        sliding_window=(window_size, window_size),
    )
    model = FlaxLlamaForBinaryTokenClassification(
        config, rngs=rngs
    )
    byte_sequence = jax.random.randint(
        jax.random.PRNGKey(0),
        shape=(batch_size, max_position_embeddings),
        minval=0,
        maxval=256,
        dtype=jnp.uint8
    )

    # Compile and execute the model with JIT
    jit_model = jax.jit(model)
    for i in range(1):
        res = jit_model(byte_sequence)

    print(res)


if __name__ == "__main__":
    main()
