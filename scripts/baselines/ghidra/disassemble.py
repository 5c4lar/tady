import json
import os

from ghidra.program.model.listing import CodeUnit
from ghidra.util.exception import CancelledException
from ghidra.program.model.block import BasicBlockModel

# Get the current program
path = currentProgram.getExecutablePath()
# Get the output file path from the script arguments
args = getScriptArgs()
if len(args) > 0:
    base_dir = args[0]
    output_dir = args[1]
else:
    raise Exception("Output file path argument is missing")

relative_path = os.path.relpath(path, base_dir)
output_file_path = os.path.join(output_dir,(relative_path + ".json"))
parent_path = os.path.dirname(os.path.abspath(output_file_path))
if not os.path.isdir(parent_path):
    os.makedirs(parent_path)

# Get the listing (disassembly)
listing = currentProgram.getListing()

# Initialize an empty list to store addresses
instructions = []
function_entries = []
functions = []

# Iterate over the disassembled instructions
last_addr = 0
for instruction in listing.getInstructions(True):
    # Add the instruction's address to the list
    ins_addr = instruction.getAddress().getOffset() # - 0x00100000
    instructions.append(ins_addr)
    last_addr = max(last_addr, ins_addr + len(instruction.getBytes()))
    

block_model = BasicBlockModel(currentProgram)
functionManager = currentProgram.getFunctionManager()
for func in functionManager.getFunctions(True):
    func_addr = func.getEntryPoint().getOffset() # - 0x00100000
    codeBlockIterator = block_model.getCodeBlocksContaining(func.getBody(), None);
    function_entries.append(func_addr)
    function_blocks = {"entry": func_addr, "blocks":[]}
    # iter over the basic blocks
    while codeBlockIterator.hasNext(): 
        bb = codeBlockIterator.next()
        block_addr = bb.getMinAddress().getOffset() # - 0x00100000
        function_blocks["blocks"].append(block_addr)
    functions.append(function_blocks)
    

start = min(instructions)
end = last_addr
    
results = {"instructions": sorted(instructions), "start": start, "end": end, "functions": function_entries, "blocks": functions}


# Write the JSON output to the file
print(output_file_path)
with open(output_file_path, "w") as output_file:
    json.dump(results, output_file)