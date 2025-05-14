from proto.blocks_pb2 import Module

'''
for (auto &func: blocks_gt.functions()) {
    functions.insert(func.va());
    for (auto &block: func.basic_blocks()) {
        blocks.insert(block.va());
        for (auto &instr: block.instructions()) {
            instructions.insert(instr.va());
        }
    }
}
'''
def parse_pb(pb):
    module = Module()
    with open(pb, "rb") as f:
        module.ParseFromString(f.read())
    instructions = []
    function_entries = []
    functions = []
    end = 0
    for func in module.functions:
        function_entries.append(func.va)
        func_blocks = {"entry": func.va, "blocks":[]}
        for block in func.basic_blocks:
            func_blocks["blocks"].append(block.va)
            for instr in block.instructions:
                instructions.append(instr.va)
                end = max(end, instr.va + instr.size)
        functions.append(func_blocks)
    # start = min(instructions)
    if len(instructions) == 0:
        start = 0
    else:
        start = min(instructions)
    return {"instructions": sorted(instructions), "start": start, "end": end, "functions": function_entries, "blocks": functions}
        
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--pb", required=True, help="Path to the protobuf file")
    args = parser.parse_args()
    res = parse_pb(args.pb)
    print(res)
    
if __name__ == "__main__":
    main()