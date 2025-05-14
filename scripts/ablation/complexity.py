#%%
from tady.model.tady_flax import *
from flax import nnx
import jax
import jax.numpy as jnp

def get_cost(window_size, seq_len, attention_type):
    config = TAGNNConfig(
        attention_type=attention_type,
        sliding_window=(window_size, window_size),
        max_position_embeddings=seq_len
    )
    rngs = nnx.Rngs(params=jax.random.key(
        0), dropout=jax.random.key(1), carry=jax.random.key(2))
    if attention_type == "full":
        attn = FlaxFullAttention(config, rngs=rngs)
    elif attention_type == "selective":
        attn = FlaxSelectiveAttention(config, rngs=rngs)
    else:
        attn = FlaxSlidingWindowAttention(config, rngs=rngs)
    batch_size = 1
    seq_len = config.max_position_embeddings
    d_model = config.hidden_size

    hidden_states =  jax.numpy.ones((batch_size, seq_len, d_model))
    position_ids = jax.numpy.broadcast_to(jnp.arange(seq_len, dtype=jnp.int32)[
        None, :], (batch_size, seq_len))
    connections = jnp.ones((batch_size, seq_len, config.num_global_connections), dtype=jnp.int32)
    if attention_type == "full":
        m = jax.jit(attn).lower(hidden_states, None, position_ids).compile()
    elif attention_type == "selective":
        m = jax.jit(attn).lower(hidden_states, connections, position_ids).compile()
    else:
        attention_mask = jnp.ones((batch_size, seq_len, 1, sum(config.sliding_window) + 1), dtype=jnp.bool)
        m = jax.jit(attn).lower(hidden_states, attention_mask, position_ids).compile()
    
    return m.cost_analysis()
    
# %%
# window_sizes = [16, 32, 64, 128, 256, 512, 1024]
seq_lens = [64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768]
attention_types = ["full", "lite", "selective"]
flops = {"full": [], "lite": [], "selective": []} # type: ignore
window_size = 64
for seq_len in seq_lens:
    for attention_type in attention_types:
        # print(f"Window size: {window_size}, Attention type: {attention_type}")
        flops[attention_type].append(get_cost(window_size, seq_len, attention_type)["flops"])
# %%
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 5))
plt.plot(seq_lens, flops["full"], label="Full attention")
plt.plot(seq_lens, flops["lite"], label="Sliding window attention")
plt.plot(seq_lens, flops["selective"], label="Selective attention")
plt.legend()
plt.xlabel("Sequence length")
plt.ylabel("FLOPs")
plt.title(f"Flops scaling for full variants of attention")
# plt.show()
plt.savefig("artifacts/attention_variants.pdf")

# %%
seq_lens_sliding_window = [512, 2048, 8192, 32768]
window_sizes = [4, 8, 16, 32, 64, 128, 256, 512, 1024]
flops_window_size = {w: [] for w in window_sizes} # type: ignore
for w in window_sizes:
    for seq_len in seq_lens_sliding_window:
        # print(f"Window size: {window_size}, Sequence length: {seq_len}")
        flops_window_size[w].append(get_cost(w, seq_len, "lite")["flops"])
    

# %%
plt.figure(figsize=(10, 5))
for window_size in window_sizes:
    plt.plot(seq_lens_sliding_window, flops_window_size[window_size], label=f"Window size: {window_size}")
plt.legend()
plt.xlabel("Sequence length")
plt.ylabel("FLOPs")
plt.title(f"Flops scaling of different window sizes")
# plt.show()
plt.savefig("artifacts/sliding_window_size.pdf")
# %%
