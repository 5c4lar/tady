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

from tady.model.attention import get_attention_wccs, get_attention_lite
from tady.model.disasm_jax import DisasmJax, byte_sequence_to_instr_bytes, overlapping_mask

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
        },
        # overlapping_attn: bool = False,
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
        attention_type="lite",
        successor_idx=(0, 3),
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
        # self.overlapping_attn = overlapping_attn
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
        self.successor_idx = successor_idx
        self.mlp_bias = mlp_bias

        self.sliding_window = sliding_window

        self.token_pool = token_pool

        # Validate the correctness of rotary position embeddings parameters
        # BC: if there is a 'type' field, move it to 'rope_type'.
        if self.rope_scaling is not None and "type" in self.rope_scaling:
            self.rope_scaling["rope_type"] = self.rope_scaling["type"]
        rope_config_validation(self)
        super().__init__(**kwargs)


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


def create_sinusoidal_positions(num_pos, dim, dtype=jnp.float32):
    inv_freq = 1.0 / \
        (10000 ** (jnp.arange(0, dim, 2, dtype=jnp.int32) / dim))  # (dim // 2)
    freqs = jnp.einsum("i , j -> i j", jnp.arange(num_pos, dtype=jnp.int32),
                       inv_freq).astype(dtype)  # (num_pos, dim // 2)

    emb = jnp.concatenate((freqs, freqs), axis=-1)  # (num_pos, dim)
    out = jnp.concatenate(
        # (num_pos, 1, 2 * dim)
        (jnp.sin(emb)[:, None, :], jnp.cos(emb)[:, None, :]), axis=-1)
    return out


def rotate_half(tensor):
    """Rotates half the hidden dims of the input."""
    rotate_half_tensor = jnp.concatenate(
        (-tensor[..., tensor.shape[-1] // 2:], tensor[..., : tensor.shape[-1] // 2]), axis=-1
    )
    return rotate_half_tensor


def apply_rotary_pos_emb(tensor, sin_pos, cos_pos):
    return (tensor * cos_pos) + (rotate_half(tensor) * sin_pos)


class FlaxLlamaRotaryEmbedding(nnx.Module):
    def __init__(self, config: TAGNNConfig,
                 dtype: jnp.dtype = jnp.float32, *, rngs: nnx.Rngs):
        self.config = config
        self.dtype = dtype
        self.head_dim = self.config.hidden_size // self.config.num_attention_heads
        num_positions = max(sum(config.sliding_window) + 1, 29) if config.attention_type != "full" else config.max_position_embeddings
        self.sincos = nnx.Variable(create_sinusoidal_positions(
            # (num_positions, 1, 2 * head_dim)
            num_positions, self.head_dim, dtype=self.dtype))

    def __call__(self, key, query, position_ids):
        # (batch_size, seq_len, 1, 2 * head_dim)
        sincos = self.sincos[position_ids]
        # (batch_size, seq_len, 1, head_dim)
        sin_pos, cos_pos = jnp.split(sincos, 2, axis=-1)
        # (batch_size, seq_len, num_heads, head_dim)
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


class FlaxFullAttention(nnx.Module):

    def __init__(self, config: TAGNNConfig,
                 dtype: jnp.dtype = jnp.float32, *, rngs: nnx.Rngs):
        self.config = config
        self.dtype = dtype
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
        rngs: Optional[nnx.Rngs] = None,
        deterministic: bool = True,
        output_attentions: bool = False,
    ):
        batch_size, seq_len = hidden_states.shape[:2]
        position_ids = jnp.broadcast_to(jnp.arange(seq_len, dtype=jnp.int32)[
            None, :], (batch_size, seq_len))
        query = self.q_proj(hidden_states)
        key = self.k_proj(hidden_states)
        value = self.v_proj(hidden_states)

        query = self._split_heads(query, self.num_heads)
        key = self._split_heads(key, self.num_key_value_heads)
        value = self._split_heads(value, self.num_key_value_heads)
        key, query = self.rotary_emb(key, query, position_ids)

        dropout_rng = None
        if not deterministic and rngs is not None and self.config.attention_dropout > 0.0:
            dropout_rng = rngs.dropout()

        attention_dtype = jnp.float32 if self.attention_softmax_in_fp32 else self.dtype
        attn_output = nnx.dot_product_attention(
            query, key, value,
            dropout_rng=dropout_rng,
            dropout_rate=self.config.attention_dropout,
            deterministic=deterministic,
            dtype=attention_dtype)
        # attn_output = jnp.einsum("...hqk,...khd->...qhd", attn_weights, value)
        attn_output = self._merge_heads(attn_output)
        attn_output = self.o_proj(attn_output)

        outputs = (attn_output,)
        return outputs


class FlaxSlidingWindowAttention(nnx.Module):

    def __init__(self, config: TAGNNConfig, sliding_window: Tuple[int, int],
                 dtype: jnp.dtype = jnp.float32, *, rngs: nnx.Rngs):
        self.config = config
        self.dtype = dtype
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.attention_softmax_in_fp32 = self.dtype is not jnp.float32
        self.sliding_window = sliding_window

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
        rngs: Optional[nnx.Rngs] = None,
        deterministic: bool = True,
        output_attentions: bool = False,
    ):
        # (batch_size, seq_len, hidden_size)
        query = self.q_proj(hidden_states)
        key = self.k_proj(hidden_states)
        value = self.v_proj(hidden_states)

        # (batch_size, seq_len, num_heads, head_dim)
        query = self._split_heads(query, self.num_heads)
        # (batch_size, seq_len, num_key_value_heads, head_dim)
        key = self._split_heads(key, self.num_key_value_heads)
        # (batch_size, seq_len, num_key_value_heads, head_dim)
        value = self._split_heads(value, self.num_key_value_heads)
        batch_size, seq_len = hidden_states.shape[:2]

        dropout_rng = None
        if not deterministic and rngs is not None and self.config.attention_dropout > 0.0:
            dropout_rng = rngs.dropout()

        # (batch_size, seq_len, num_heads, head_dim)
        key = jnp.repeat(key, self.num_key_value_groups, axis=2)
        value = jnp.repeat(value, self.num_key_value_groups, axis=2)
        key, value = map(lambda t: jnp.pad(
            t, ((0, 0), self.sliding_window, (0, 0), (0, 0))), (key, value))
        window_size = sum(self.sliding_window) + 1
        idx = jnp.arange(window_size, dtype=jnp.int32)[
            # (seq_len, window_size)
            None, :] + jnp.arange(seq_len, dtype=jnp.int32)[:, None]
        # (batch_size, seq_len, window_size, num_heads, head_dim)
        key, value = map(lambda t: t[:, idx, :, :], (key, value))
        # (batch_size, seq_len, 1, num_heads, head_dim)
        query = jnp.expand_dims(query, 2)

        position_ids = jnp.broadcast_to(jnp.arange(window_size, dtype=jnp.int32)[
            None, :], (batch_size * seq_len, window_size))
        key, value = map(lambda t: t.reshape(
            batch_size * seq_len, window_size, self.num_heads, self.head_dim), (key, value))
        key, value = self.rotary_emb(key, value, position_ids)
        key, value = map(lambda t: t.reshape(
            batch_size, seq_len, window_size, self.num_heads, self.head_dim), (key, value))

        attention_variants = attention_mask.shape[2] #repeat the attention mask to match the number of heads
        num_repeat = self.config.num_attention_heads // attention_variants
        attention_mask = jnp.repeat(attention_mask[:, :, :, None, :], num_repeat, axis=2)
        # attention_mask = jnp.broadcast_to(
        #     attention_mask[:, :, None, :, :],
        #     # (batch_size, seq_len, num_heads, 1, window_size)
        #     (batch_size, seq_len, self.config.num_attention_heads, 1, window_size))
        # usual dot product attention
        attention_dtype = jnp.float32 if self.attention_softmax_in_fp32 else self.dtype
        # transform boolean mask into float mask
        # attention_bias = lax.select(
        #     attention_mask > 0,
        #     jnp.full(attention_mask.shape, 0.0).astype(self.dtype),
        #     jnp.full(attention_mask.shape, jnp.finfo(
        #         self.dtype).min).astype(self.dtype),
        # )
        attn_output = nnx.dot_product_attention(
            query, key, value,
            # bias=attention_bias,
            mask=attention_mask,
            dropout_rng=dropout_rng,
            dropout_rate=self.config.attention_dropout,
            deterministic=deterministic,
            # （batch_size, seq_len, num_heads , head_dim)
            dtype=attention_dtype).squeeze(2)
        # attn_output = jnp.einsum("...hqk,...khd->...qhd", attn_weights, value)
        # (batch_size, seq_len, hidden_size)
        attn_output = self._merge_heads(attn_output)
        # (batch_size, seq_len, hidden_size)
        attn_output = self.o_proj(attn_output)

        outputs = (attn_output,)
        return outputs


class FlaxSelectiveAttention(nnx.Module):

    def __init__(self, config: TAGNNConfig,
                 dtype: jnp.dtype = jnp.float32, *, rngs: nnx.Rngs):
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
            self.k_projs = {}
            self.v_projs = {}
            for key, (a, b) in self.config.connections.items():
                self.k_projs[key] = nnx.Linear(
                    in_features=self.config.hidden_size,
                    out_features=self.num_key_value_heads * self.head_dim,
                    use_bias=config.attention_bias,
                    dtype=self.dtype,
                    param_dtype=self.dtype,
                    kernel_init=jax.nn.initializers.normal(
                        self.config.initializer_range, dtype=self.dtype),
                    rngs=rngs
                )
                self.v_projs[key] = nnx.Linear(
                    in_features=self.config.hidden_size,
                    out_features=self.num_key_value_heads * self.head_dim,
                    use_bias=config.attention_bias,
                    dtype=self.dtype,
                    param_dtype=self.dtype,
                    kernel_init=jax.nn.initializers.normal(
                        self.config.initializer_range, dtype=self.dtype),
                    rngs=rngs
                )
        else:
            self.k_projs_all = nnx.Linear(
                in_features=self.config.hidden_size,
                out_features=self.num_key_value_heads * self.head_dim,
                use_bias=config.attention_bias,
                dtype=self.dtype,
                param_dtype=self.dtype,
                kernel_init=jax.nn.initializers.normal(
                    self.config.initializer_range, dtype=self.dtype),
                rngs=rngs
            )  # type: ignore
            self.v_projs_all = nnx.Linear(
                in_features=self.config.hidden_size,
                out_features=self.num_key_value_heads * self.head_dim,
                use_bias=config.attention_bias,
                dtype=self.dtype,
                param_dtype=self.dtype,
                kernel_init=jax.nn.initializers.normal(
                    self.config.initializer_range, dtype=self.dtype),
                rngs=rngs
            )  # type: ignore

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
        # self.rotary_emb = FlaxLlamaRotaryEmbedding(
        #     config, dtype=self.dtype, rngs=rngs)

    def _split_heads(self, hidden_states, num_heads):
        return hidden_states.reshape(hidden_states.shape[:3] + (num_heads, self.head_dim))

    def _merge_heads(self, hidden_states):
        return hidden_states.reshape(hidden_states.shape[:2] + (self.embed_dim,))

    def __call__(
        self,
        hidden_states,
        connections,
        rngs: Optional[nnx.Rngs] = None,
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
        if self.config.global_connection_class:
            key = jnp.concatenate([self.k_projs[key](connected_hidden_states[:, :, a: b, :])
                                   for key, (a, b) in self.config.connections.items()], axis=2)
            value = jnp.concatenate([self.v_projs[key](connected_hidden_states[:, :, a: b, :])
                                     for key, (a, b) in self.config.connections.items()], axis=2)
        else:
            key = self.k_projs_all(connected_hidden_states)
            value = self.v_projs_all(connected_hidden_states)
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
        if not deterministic and rngs and self.config.attention_dropout > 0.0:
            dropout_rng = rngs.dropout()  # type: ignore

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
        if self.config.attention_type == "full":
            self.self_attn = FlaxFullAttention(
                self.config, dtype=self.dtype, rngs=rngs)
        else:
            self.self_attn = FlaxSlidingWindowAttention(
                self.config, self.config.sliding_window, dtype=self.dtype, rngs=rngs)  # type: ignore
        if self.config.num_global_connections != 0:
            self.selective_attn = FlaxSelectiveAttention(
                self.config, dtype=self.dtype, rngs=rngs)
        # if self.config.overlapping_attn:
        #     self.overlapping_attn = FlaxSlidingWindowAttention(
        #         self.config, (14, 14), dtype=self.dtype, rngs=rngs)
        self.post_attention_layernorm = FlaxLlamaRMSNorm(
            self.config, dtype=self.dtype, rngs=rngs)
        self.mlp = FlaxLlamaMLP(self.config, dtype=self.dtype, rngs=rngs)

    def __call__(
        self,
        hidden_states,
        attention_mask=None,
        connections=None,
        # overlappings=None,
        rngs: Optional[nnx.Rngs] = None,
        deterministic: bool = True,
        output_attentions: bool = False,
    ):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        outputs = self.self_attn(
            hidden_states,
            attention_mask=attention_mask,
            rngs=rngs,
            deterministic=deterministic,
            output_attentions=output_attentions,
        )
        # residual connection
        attn_output = outputs[0]
        if self.config.num_global_connections != 0:
            outputs = self.selective_attn(
                hidden_states,
                connections,
                rngs=rngs,
                deterministic=deterministic,
            )
            selective_output = outputs[0]
            residual = residual + selective_output
        # if self.config.overlapping_attn:
        #     outputs = self.overlapping_attn(
        #         hidden_states,
        #         overlappings,
        #         rngs=rngs,
        #         deterministic=deterministic,
        #     )
        #     residual = residual + outputs[0]
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
        connections=None,
        # overlappings=None,
        rngs: Optional[nnx.Rngs] = None,
        deterministic: bool = True,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
    ):
        all_attentions = () if output_attentions else None
        all_hidden_states = () if output_hidden_states else None

        for block in self.blocks:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)  # type: ignore
            layer_outputs = block(
                hidden_states,
                attention_mask=attention_mask,
                connections=connections,
                # overlappings=overlappings,
                rngs=rngs,
                deterministic=deterministic,
                output_attentions=output_attentions,
            )
            hidden_states = layer_outputs[0]

            if output_attentions:
                all_attentions += (layer_outputs[1],)  # type: ignore

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
        self.mode_embedding = nnx.Embed(
            2,
            self.hidden_size,
            embedding_init=embedding_init,
            dtype=self.dtype,
            param_dtype=self.dtype,
            rngs=rngs,
        )
        self.invalid_embedding = nnx.Param(
            jax.random.normal(
                rngs.params(), (self.hidden_size,), dtype=self.dtype)
        )
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
                activation_fn=ACT2FN[config.hidden_act],  # type: ignore
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
        connections=None,
        instr_len=None,
        is_64=None,
        rngs: Optional[nnx.Rngs] = None,
        deterministic=True,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ):
        input_embeds = self.embed_tokens(input_ids.astype("i4"))
        if instr_len is None:
            instr_len = jnp.ones(
                (input_ids.shape[0], input_ids.shape[1])) * input_ids.shape[2]
        input_embeds = jnp.where(
            instr_len[:, :, None, None] != 0, input_embeds, self.invalid_embedding[None, None, None, :])
        instr_len = jnp.where(
            instr_len != 0, instr_len, 1
        )
        is_64 = is_64 if is_64 is not None else jnp.ones(
            input_ids.shape[0], dtype=jnp.bool_)
        mode_embeds = self.mode_embedding(is_64.astype("i4"))
        input_embeds = input_embeds + mode_embeds[:, None, None, :]
        if self.config.attention_type == "graph":
            attention_mask = jax.vmap(get_attention_wccs, in_axes=(0, None))(
                connections[..., :3], self.config.sliding_window
            )
        elif self.config.attention_type == "lite":
            # attention_mask_must = jax.vmap(get_attention_lite, in_axes=(0, None))(
            #     connections[..., 0], self.config.sliding_window) 
            # attention_mask_next = jax.vmap(get_attention_lite, in_axes=(0, None))(
            #     connections[..., 3], self.config.sliding_window)
            attention_masks = [
                jax.vmap(get_attention_lite, in_axes=(0, None))(
                    connections[..., i], self.config.sliding_window)
                for i in self.config.successor_idx
            ]
            overlappings = jax.vmap(overlapping_mask)(instr_len) # (batch_size, seq_len, 1, 29)
            overlappings = jnp.pad(overlappings, ((0, 0), (0, 0), (0, 0), (self.config.sliding_window[0] - 14, self.config.sliding_window[1] - 14)), mode='empty') # (batch_size, seq_len, 1, window_size)
            attention_masks.append(overlappings)
            # add full attention mask to fill the shape to (batch_size, seq_len, num_heads, window_size)
            
            attention_mask = jnp.concatenate(
                attention_masks, axis=-2) # (batch_size, seq_len, 3, window_size)
            attention_mask = jnp.pad(attention_mask, ((0, 0), (0, 0), (0, self.config.num_attention_heads - attention_mask.shape[-2]), (0, 0)), constant_values=True) # (batch_size, seq_len, 3, window_size)
        elif self.config.attention_type == "sliding":
            attention_mask = jnp.ones(
                input_embeds.shape[:2] + (1, sum(self.config.sliding_window) + 1), dtype=jnp.bool)
        elif self.config.attention_type == "full":
            attention_mask = None
        if self.config.token_pool == "rnn":
            input_embeds = self.pool_tokens(
                jnp.reshape(input_embeds, (-1, ) + input_embeds.shape[2:]),
                initial_carry=jnp.repeat(
                    self.initial_carry[None, :], input_embeds.shape[0] * input_embeds.shape[1], axis=0),
                seq_lengths=jnp.reshape(instr_len, (-1, )),
                # rngs=rngs
            )[0].reshape(input_embeds.shape[:2] + (self.config.hidden_size,))
        elif self.config.token_pool == "mean":
            if instr_len is not None:
                instr_len_mask = jnp.arange(input_embeds.shape[2])[
                    None, None, :] < instr_len[:, :, None]
                input_embeds = jnp.sum(
                    input_embeds, axis=2, where=jnp.expand_dims(instr_len_mask, -1))

                # Compute average of valid embeddings
                input_embeds = input_embeds / (instr_len[:, :, None] + 1e-9)
            else:
                input_embeds = jnp.mean(input_embeds, axis=2)
        elif self.config.token_pool == "sum":
            input_embeds = jnp.sum(
                input_embeds, axis=2)

        # if self.config.overlapping_attn:
        #     overlappings = jax.vmap(overlapping_mask)(instr_len)
        # else:
        #     overlappings = None
        outputs = self.layers(
            input_embeds,
            attention_mask=attention_mask,
            connections=connections,
            # overlappings=overlappings,
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
        connections=None,
        instr_len=None,
        is_64=None,
        rngs: Optional[nnx.Rngs] = None,
        deterministic=True,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ):
        batch_size, seq_len = input_ids.shape[:2]
        outputs = self.model(
            input_ids,
            instr_len=instr_len,
            is_64=is_64,
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
        is_64: Optional[jnp.ndarray] = None,
        rngs: Optional[nnx.Rngs] = None,
        deterministic=True,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
        **kwargs,
    ):
        if is_64 is None:
            is_64 = jnp.ones(
                byte_sequence.shape[0], dtype=jnp.bool_)
        batch_size, seq_len = byte_sequence.shape[:2]
        input_ids, connections, instr_len, _ = nnx.vmap(
            self.disassembler, in_axes=(0, 0))(byte_sequence, is_64)
        outputs = self.model(
            input_ids,
            instr_len=instr_len,
            is_64=is_64,
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


class XDA(nnx.Module):
    def __init__(self, config: TAGNNConfig,
                 dtype: jnp.dtype = jnp.float32, *, rngs: nnx.Rngs):
        self.config = config
        assert self.config.num_global_connections == -1
        assert self.config.attention_type == "full"
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
        byte_sequence: jnp.ndarray,
        is_64: Optional[jnp.ndarray] = None,
        rngs: Optional[nnx.Rngs] = None,
        deterministic=True,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
        **kwargs,
    ):
        batch_size, seq_len = byte_sequence.shape[:2]
        input_ids = byte_sequence.astype(
            jnp.int32).reshape((batch_size, seq_len, 1))
        outputs = self.model(
            input_ids,
            instr_len=None,
            is_64=is_64,
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

class Tady(nnx.Module):
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
        byte_sequence: jnp.ndarray,
        is_64: jnp.ndarray,
        instr_len: jnp.ndarray,
        control_flow: jnp.ndarray,
        rngs: Optional[nnx.Rngs] = None,
        deterministic=True,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
        **kwargs,
    ):
        batch_size, seq_len = byte_sequence.shape[:2]
        input_ids = jax.vmap(byte_sequence_to_instr_bytes)(byte_sequence)
        idx = jnp.arange(0, input_ids.shape[2])
        input_ids += idx[None, None, :] * 256
        # calculate overlapping connections based on instr_len
        # overlapping = jax.vmap(overlapping_addresses)(instr_len)
        # overlapping_prev = jax.vmap(overlapping_addresses_prev)(instr_len)
        # connections = jnp.concatenate(
        #     [control_flow, overlapping, overlapping_prev], axis=-1)
        outputs = self.model(
            input_ids,
            instr_len=instr_len,
            is_64=is_64,
            connections=control_flow,
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
    jax.config.update("jax_debug_nans", True)

    num_attention_heads = 4
    hidden_size = 32
    intermediate_size = 64
    window_size = 32
    batch_size = 32
    max_position_embeddings = 8192
    node_size = 40
    num_neighbors = 1
    num_conflicts = 14
    num_embeddings = 256 * 15
    num_hidden_layers = 4
    rngs = nnx.Rngs(params=jax.random.key(
        0), dropout=jax.random.key(1), carry=jax.random.key(2))
    config = TAGNNConfig(
        num_attention_heads=num_attention_heads,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        window_size=window_size,
        max_position_embeddings=max_position_embeddings,
        vocab_size=num_embeddings,
        num_hidden_layers=num_hidden_layers,
        attention_type="sliding",
        token_pool="sum",
        sliding_window=(window_size, window_size),
    )
    model = FlaxLlamaForBinaryTokenClassification(
        config, rngs=rngs
    )
    from tady.utils.loader import preprocess_binary
    byte_chunks, masks, use_64_bit, base_addr = preprocess_binary(
        "/bin/bash", section_name=".text")

    print(byte_chunks.shape)
    res = model(byte_chunks[:32])
    # Compile and execute the model with JIT
    # jit_model = jax.jit(model)
    # for i in range(1):
    #     res = jit_model(byte_sequence)

    print(res)


if __name__ == "__main__":
    main()
