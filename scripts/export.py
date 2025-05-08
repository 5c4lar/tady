import pathlib

import hydra
import jax
import jax.numpy as jnp
import orbax.checkpoint
import tensorflow as tf
from flax import nnx
from jax.experimental import jax2tf
from omegaconf import DictConfig
from tady.model.tagnn_flax import *


@hydra.main(version_base=None, config_path="conf", config_name="export")
def main(args: DictConfig):
    model_id = "_".join([str(i) for i in args.tags])
    checkpoint_path = pathlib.Path(args.checkpoint) / model_id
    config_path = pathlib.Path(args.config) / model_id
    output_path = pathlib.Path(args.output) / model_id / '1'
    print(f"Exporting {checkpoint_path.absolute()}")
    config = TAGNNConfig.from_pretrained(config_path)
    rngs = nnx.Rngs(params=jax.random.key(
        0), dropout=jax.random.key(1), carry=jax.random.key(2))
    match (args.model.dtype):
        case "float32":
            dtype = jnp.float32
        case "bfloat16":
            dtype = jnp.bfloat16
        case "float16":
            dtype = jnp.float16
    config.sliding_window = tuple(config.sliding_window)
    model = FlaxLlamaForBinaryTokenClassification(
        config,
        dtype=dtype,
        rngs=rngs
    )
    checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    ckpt_path = pathlib.Path(checkpoint_path)
    state = checkpointer.restore(ckpt_path.absolute(), nnx.state(model))
    nnx.update(model, state)

    graphdef, statedef = nnx.split(model)
    pure_state = statedef.to_pure_dict()

    @jax.jit
    def forward_jax(pure_state, byte_sequence, use_64_bit):
        '''
        Forward pass for the model. This is used for exporting the model to TensorFlow.
        Args:
            pure_state: The pure state of the model.
            byte_sequence: The input sequence.
            use_64_bit: Whether to disassemble the byte sequence as 64 bit.
        Returns:
            The output of the model.
        '''
        state = jax.tree.map(lambda x: x, statedef)  # copy
        state.replace_by_pure_dict(pure_state)
        model = nnx.merge(graphdef, state)
        output = model(byte_sequence, use_64_bit,
                       deterministic=True).logits.squeeze(-1)
        return nnx.sigmoid(output).astype(jnp.float32)

    m = tf.Module()

    def convert_jax_value(x):
        if x is None:
            return None
        if isinstance(x, jax._src.prng.PRNGKeyArray):
            return None
        return tf.Variable(x)

    tf_state = tf.nest.map_structure(convert_jax_value, pure_state)
    m.vars = tf.nest.flatten(tf_state)
    tf_args = [tf.TensorSpec([None, None], tf.uint8, name="byte_sequence"),
               tf.TensorSpec([None,], tf.bool, name="use_64_bit")
               ]

    @tf.function(autograph=False, input_signature=tf_args, jit_compile=True)
    def predict_tf(byte_sequence, use_64_bit):
        return jax2tf.convert(forward_jax, polymorphic_shapes=[
            "...",
            "(a, b)",
            "(a,)"
        ], with_gradient=False,
        )(tf_state, byte_sequence, use_64_bit)

    m.predict = predict_tf
    tf.saved_model.save(m, output_path)


if __name__ == "__main__":
    main()
