import numpy as np
from fairseq.models.roberta import RobertaModel
import argparse
import json
import pathlib
import tqdm
from multiprocessing import Pool
import torch
import lief

def load_text(file_path, section_name=None):
    binary = lief.parse(str(file_path))

    if isinstance(binary, lief.ELF.Binary):
        name = ".text" if section_name is None else section_name
        text_section = binary.get_section(name)
        base_addr = text_section.virtual_address
        text_bytes = bytes(text_section.content)
        use_64_bit = binary.header.identity_class == lief.ELF.ELF_CLASS.CLASS64

    elif isinstance(binary, lief.MachO.Binary):
        name = "__text" if section_name is None else section_name
        text_section = binary.get_section(name)
        base_addr = text_section.virtual_address
        text_bytes = bytes(text_section.content)
        use_64_bit = binary.header.cpu_type == lief.MachO.CPU_TYPES.X86_64

    elif isinstance(binary, lief.PE.Binary):
        name = ".text" if section_name is None else section_name
        text_section = binary.get_section(name)
        base_addr = text_section.virtual_address + binary.optional_header.imagebase
        text_bytes = bytes(text_section.content)
        use_64_bit = binary.header.machine == lief.PE.MACHINE_TYPES.AMD64
    else:
        raise ValueError(f"Unsupported binary type: {type(binary)}")
    return text_bytes, use_64_bit, base_addr


class XDA:
    def __init__(self, gpu, batch_size, model_path, dict_path):
        self.disasm = RobertaModel.from_pretrained(model_path, 'checkpoint_best.pt', dict_path,
                                                    bpe=None, user_dir='finetune_tasks')
        if gpu:
            self.disasm = self.disasm.to('cuda')
        self.disasm.eval()
        self.batch_size = batch_size

    def disassemble(self, path, section_name):
        code, use_64_bit, base_addr = load_text(path, section_name)

        inst_pred = np.zeros(len(code), dtype=np.bool_)
        logits = np.zeros(len(code), dtype=np.float32)
        for i in range(0, len(code), self.batch_size):
            # print(f"Processing: {code[i:i+self.batch_size]}" )
            batch = code[i:i+self.batch_size]
            tokens = " ".join(batch[j:j+1].hex() for j in range(len(batch)))
            encoded_tokens = self.disasm.encode(tokens).to('cuda')
            with torch.no_grad():
                logprobs = self.disasm.predict('instbound', encoded_tokens)
            preds = (logprobs.argmax(dim=2).view(-1)
                         == 1).detach().cpu().numpy()
            probs = torch.softmax(logprobs, dim=2)[
                0, ..., 1].detach().cpu().numpy()
            inst_pred[i:i+self.batch_size] = preds
            logits[i:i+self.batch_size] = -np.log((1 / (probs + 1e-8)) - 1)
        return inst_pred, logits, base_addr


def process_file(args, file, xda):
    try:
        print(f"Processing {file}")
        if file.is_file():
            rel_path = file.relative_to(args.dir)
            output_path = pathlib.Path(args.output) / (str(rel_path) + ".npz")
            if output_path.exists():
                return
            print(f"processing {file}")
            pred, logits, base_addr = xda.disassemble(file, args.section_name)
            result = {
                "pred": pred.astype(np.bool_),
                "logits": logits.astype(np.float32),
                "base_addr": np.array(base_addr, dtype=np.uint64)
            }
            output_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez(output_path, **result)
    except Exception as e:
        print(e)
        return


def main():
    parser = argparse.ArgumentParser(description='XDA example')
    parser.add_argument('--file', type=str, help='File to process')
    parser.add_argument('--gpu', action='store_true',
                        help='Enable GPU acceleration')
    parser.add_argument(
        '--dir', help='Path to the directory of binary to disassemble', required=False)
    parser.add_argument(
        '--output', help='Path to the directory to store output', required=False)
    parser.add_argument(
        '--scores', help='Path to the directory to store the scores', required=False)
    parser.add_argument("--process", type=int,
                        help="Num of process used to process the stats")
    parser.add_argument("--model_path", default="checkpoints/finetune_instbound_new_dataset", type=str,
                        help="Path to the model")
    parser.add_argument("--dict_path", default="xda_dataset/processed", type=str,
                        help="Path to the dict")
    parser.add_argument("--section_name", type=str, help="Section name to disassemble")
    args = parser.parse_args()
    print("Starting...")
    if args.file:
        xda = XDA(args.gpu, 512, args.model_path, args.dict_path)
        process_file(args, pathlib.Path(args.file), xda)
    else:
        root_dir = pathlib.Path(args.dir)
        xda = XDA(args.gpu, 512, args.model_path, args.dict_path)
        files = list([i for i in root_dir.rglob("*") if i.is_file()])  # Convert generator to list

        for file in tqdm.tqdm(files):
            process_file(args, file, xda)
        # pool = Pool(args.process)

        # tasks = [(args, file) for file in files if file.is_file()]
        # with Pool(args.process) as pool, tqdm.tqdm(total=len(files)) as pbar:
        #     for result in pool.imap_unordered(process_file, tasks):
        #         pbar.update()
        #         pbar.refresh()


if __name__ == '__main__':
    main()
