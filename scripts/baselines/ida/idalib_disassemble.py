import re
import json
import shutil
import tempfile
import pathlib

def parse_binary(file, persist_database):
    import idapro
    import idautils
    import idaapi
    import idc
    try:
        with tempfile.TemporaryDirectory(dir="/dev/shm") as d:
            dst_path = pathlib.Path(d) / file.name
            shutil.copy(file, dst_path)
            idapro.open_database(str(dst_path), True)
            instructions = set()
            function_entries = []
            functions = []
            last_addr = 0
            for func_ea in idautils.Functions():
                # Retrieve the function object
                func = idaapi.get_func(func_ea)
                if not func:
                    continue
                function_entries.append(func_ea)
                function_blocks = {"entry": func_ea, "blocks":[]}
                # Get the basic blocks within the function
                flow = idaapi.FlowChart(func)
                for block in flow:
                    function_blocks["blocks"].append(block.start_ea)
                    ea = block.start_ea
                    while ea < block.end_ea and ea >= block.start_ea:
                        instructions.add(ea)
                        ea = idc.next_head(ea, block.end_ea)
                        if ea == idc.BADADDR:
                            continue
                        last_addr = max(last_addr, ea)
                functions.append(function_blocks)
            start = min(instructions)
            end = last_addr
            idapro.close_database(save=persist_database)
            results = {"instructions": sorted(list(instructions)), "start": start, "end": end, "functions": function_entries, "blocks": functions}
            return results
    except Exception as e:
        print(e.with_traceback())
        return None

def main():
    import argparse
    parser=argparse.ArgumentParser(description="IDA Python Library Demo")
    parser.add_argument("-f", "--file", help="File to be analyzed with IDA", type=str, required=True)
    parser.add_argument("-p", "--persist-database", help="Persist database changes", action='store_true')
    args=parser.parse_args()
    res = parse_binary(pathlib.Path(args.file), args.persist_database)
    print(res)

if __name__ == "__main__":
    main()