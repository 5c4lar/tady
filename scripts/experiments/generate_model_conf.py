import pathlib

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dir')
args = parser.parse_args()

directories = list(i.name for i in pathlib.Path(args.dir).glob("*") if i.is_dir())

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
with open(pathlib.Path(args.dir)/"model.conf", 'w') as f:
    f.write(base_template.format(content=content))
    
with open(pathlib.Path(args.dir)/"batching.conf", 'w') as f:
    f.write(batching_conf)