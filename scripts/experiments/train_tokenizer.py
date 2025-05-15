import os
os.environ["TOKENIZERS_PARALLELISM"] = "true"
from tokenizers import Tokenizer, models, pre_tokenizers, trainers
import itertools
import datasets
from tady import cpp
from omegaconf import DictConfig
import hydra
import random
def asms_iterator(ds):
    for item in ds:
        bytes = item['bytes']
        is_64 = item['is_64']
        instrs = cpp.Disassembler().disasm_to_str(bytes, is_64, 0)
        yield instrs

@hydra.main(version_base=None, config_path="conf", config_name="tokenizer")
def main(args: DictConfig):
    tokenizer = Tokenizer(models.BPE())
    tokenizer.enable_padding()
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    # tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size = 2047,
        show_progress = True,
    )
    ds = datasets.load_from_disk(args.dataset_dir)
    num_samples = args.num_samples if args.num_samples is not None else len(ds)
    print(f"Training from {num_samples} samples of total {len(ds)}")
    ds = ds.select(random.sample(range(len(ds)), num_samples))
    tokenizer.train_from_iterator(itertools.chain.from_iterable(asms_iterator(ds)), trainer=trainer)

    tokenizer.save(args.output)

if __name__ == "__main__":
    main()
