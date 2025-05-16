import idautils
import idaapi
import idc
import re
import json

def parse_binary():
    instructions = []
    function_entries = []
    functions = []
    last_addr = 0
    for func_ea in idautils.Functions():
        # Retrieve the function object
        func = idaapi.get_func(func_ea)
        if not func:
            continue
        function_blocks = {"entry": func_ea, "blocks":[]}
        # Get the basic blocks within the function
        flow = idaapi.FlowChart(func)
        for block in flow:
            function_blocks["blocks"].append(block.start_ea)
            ea = block.start_ea
            while ea < block.end_ea:
                instructions.append(ea)
                ea = idc.next_head(ea, block.end_ea)
                last_addr = max(last_addr, ea)
    start = min(instructions)
    end = last_addr
        
    results = {"instructions": sorted(instructions), "start": start, "end": end, "functions": function_entries, "blocks": functions}
    return results

if __name__ == '__main__':
    idc.auto_wait()
    binary_abs_path = idc.get_input_file_path()
    output_path = idc.ARGV[1]
    result = parse_binary()
    json.dump(result, open(output_path, 'w'))
    idc.qexit(0)