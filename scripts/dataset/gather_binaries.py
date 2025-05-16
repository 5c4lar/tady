import os
import shutil
import argparse
import struct

def is_macho(file_path):
    """Check if the file is a Mach-O executable or shared library by directly parsing the Mach-O header."""
    try:
        with open(file_path, 'rb') as f:
            # Read the Mach-O header
            f.seek(0)
            header = f.read(64)  # First 64 bytes contain the header
            
            # Ensure it's a Mach-O file
            if header[:4] != b'\xcf\xfa\xed\xfe':
                return False
            else:
                return True
    except Exception as e:
        return False
    
def is_pe(file_path):
    """Check if the file is a PE executable or shared library by directly parsing the PE header."""
    try:
        with open(file_path, 'rb') as f:
            # Read the PE header
            f.seek(0)
            header = f.read(64)  # First 64 bytes contain the header
            if header[:4] != b'MZ\x90\x00':
                return False
            else:
                return True
    except Exception as e:
        return False

def is_elf_executable_or_shared(file_path):
    """Check if the file is an ELF executable or shared library by directly parsing the ELF header."""
    try:
        with open(file_path, 'rb') as f:
            # Read the ELF header
            f.seek(0)
            header = f.read(64)  # First 64 bytes contain the header
            
            # Ensure it's an ELF file
            if header[:4] != b'\x7fELF':
                return False
            
            # Get the e_type field which indicates if it's executable or shared library
            # e_type is at offset 16 (0x10) and is 2 bytes
            e_type = struct.unpack('<H', header[16:18])[0]
            
            # ET_EXEC (2) is executable, ET_DYN (3) is shared object
            return e_type == 2 or e_type == 3
    except Exception as e:
        return False

def copy_binaries(src_folder, dest_folder):
    """Recursively find and copy ELF executables and shared libraries from src_folder to dest_folder."""
    for root, _, files in os.walk(src_folder):
        for file in files:
            file_path = os.path.join(root, file)
            # skip symbolic links
            if os.path.islink(file_path):
                continue
            if is_elf_executable_or_shared(file_path) or is_macho(file_path) or is_pe(file_path):
                # Create destination folder structure
                relative_path = os.path.relpath(file_path, src_folder)
                dest_path = os.path.join(dest_folder, relative_path)
                if os.path.exists(dest_path):
                    continue
                os.makedirs(os.path.dirname(dest_path), exist_ok=True)
                shutil.copy2(file_path, dest_path)
                print(f"Copied: {file_path} to {dest_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, help="Source directory to gather binaries")
    parser.add_argument("--target", type=str, help="Target directory to store binaries")
    args = parser.parse_args()
    # Usage
    src_folder = args.source
    dest_folder = args.target
    copy_binaries(src_folder, dest_folder)
    
if __name__ == "__main__":
    main()
