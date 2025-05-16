import pathlib

import hydra
import jax
import jax.numpy as jnp
import orbax.checkpoint
import tensorflow as tf
from flax import nnx
from jax.experimental import jax2tf
from omegaconf import DictConfig
from tady.model.tady_flax import *

def generate_model_conf(args: DictConfig):
    directories = list(i.name for i in pathlib.Path(args.output).glob("*") if i.is_dir())

    template = \
    """    config {{
            name: '{name}'
            base_path: '/models/{name}'
            model_platform: 'tensorflow'
        }}
    """
    # for i in directories:
    #     print(template.format(name=i))
    base_template = \
    """model_config_list {{
        {content}}}
    """

    batching_conf = '''max_batch_size { value: 128 }
    batch_timeout_micros { value: 0 }
    max_enqueued_batches { value: 1000000 }
    num_batch_threads { value: 24 }'''

    content = "".join([template.format(name=i) for i in directories])
    # print(base_template.format(content=content))
    with open(pathlib.Path(args.output)/"model.conf", 'w') as f:
        f.write(base_template.format(content=content))
        
    with open(pathlib.Path(args.output)/"batching.conf", 'w') as f:
        f.write(batching_conf)
        
def convert_jax_value(x):
    if x is None:
        return None
    if isinstance(x, jax._src.prng.PRNGKeyArray):
        return None
    return tf.Variable(x)

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
    if args.model.disassembler == "cpp":
        model = Tady(config, dtype=dtype, rngs=rngs)
    else:
        model = FlaxLlamaForBinaryTokenClassification(
            config,
            dtype=dtype,
            rngs=rngs
        )
    if args.model.disassembler == "token":
        model = FlaxLlamaForTokenClassification(
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

    if args.model.disassembler == "jax":
        @jax.jit
        def forward_jax(pure_state, byte_sequence, use_64_bit):
            state = jax.tree.map(lambda x: x, statedef)  # copy
            state.replace_by_pure_dict(pure_state)
            model = nnx.merge(graphdef, state)
            output = model(byte_sequence, use_64_bit,
                        deterministic=True).logits.squeeze(-1)
            # return nnx.sigmoid(output).astype(jnp.float32)
            return output.astype(jnp.float32)

        m = tf.Module()


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
        
    elif args.model.disassembler == "cpp":
        @jax.jit
        def forward_cpp(pure_state, byte_sequence, use_64_bit, instr_len, control_flow):
            state = jax.tree.map(lambda x: x, statedef)  # copy
            state.replace_by_pure_dict(pure_state)
            model = nnx.merge(graphdef, state)
            output = model(byte_sequence, use_64_bit, instr_len, control_flow,
                           deterministic=True).logits.squeeze(-1)
            # return nnx.sigmoid(output).astype(jnp.float32)
            return output.astype(jnp.float32)
        
        m = tf.Module()

        tf_state = tf.nest.map_structure(convert_jax_value, pure_state)
        m.vars = tf.nest.flatten(tf_state)
        tf_args = [tf.TensorSpec([None, None], tf.uint8, name="byte_sequence"),
                tf.TensorSpec([None,], tf.bool, name="use_64_bit"),
                tf.TensorSpec([None, None], tf.uint8, name="instr_len"),
                tf.TensorSpec([None, None, 4], tf.int32, name="control_flow")
                ]
        
        @tf.function(autograph=False, input_signature=tf_args, jit_compile=True)
        def predict_tf(byte_sequence, use_64_bit, instr_len, control_flow):
            return jax2tf.convert(forward_cpp, polymorphic_shapes=[
                "...",
                "(a, b)",
                "(a,)",
                "(a, b)",
                "(a, b, 4)"
            ], with_gradient=False,
            )(tf_state, byte_sequence, use_64_bit, instr_len, control_flow)

        m.predict = predict_tf
        tf.saved_model.save(m, output_path)
    elif args.model.disassembler == "token":
        @jax.jit
        def forward_token(pure_state, input_ids, connections, instr_len, is_64):
            state = jax.tree.map(lambda x: x, statedef)  # copy
            state.replace_by_pure_dict(pure_state)
            model = nnx.merge(graphdef, state)
            output = model(input_ids, connections, instr_len, is_64,
                           deterministic=True).logits.squeeze(-1)
            return output.astype(jnp.float32)
        
        m = tf.Module()
        
        tf_state = tf.nest.map_structure(convert_jax_value, pure_state)
        m.vars = tf.nest.flatten(tf_state)
        tf_args = [tf.TensorSpec([None, None, None], tf.int32, name="input_ids"),
                tf.TensorSpec([None, None, 18], tf.int32, name="connections"),
                tf.TensorSpec([None, None], tf.uint8, name="instr_len"),
                tf.TensorSpec([None,], tf.bool, name="use_64_bit")
                ]
        
        @tf.function(autograph=False, input_signature=tf_args, jit_compile=True)
        def predict_tf(input_ids, connections, instr_len, is_64):
            return jax2tf.convert(forward_token, polymorphic_shapes=[
                "...",
                "(a, b, c)",
                "(a, b, 18)",
                "(a, b)",
                "(a,)"
            ], with_gradient=False,
            )(tf_state, input_ids, connections, instr_len, is_64)

        m.predict = predict_tf
        tf.saved_model.save(m, output_path)
    generate_model_conf(args)
    print("Done!")

if __name__ == "__main__":
    main()
