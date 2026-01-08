
import os
import sys

# Path to the shared FFmpeg bin directory
ffmpeg_path = r"E:\Software\ffmpeg\bin"

# 1. Update PATH just in case
print(f"Adding {ffmpeg_path} to PATH")
os.environ["PATH"] = ffmpeg_path + os.pathsep + os.environ["PATH"]

# 2. Use add_dll_directory (Python 3.8+ Windows)
if hasattr(os, "add_dll_directory"):
    print(f"Adding {ffmpeg_path} via add_dll_directory")
    try:
        os.add_dll_directory(ffmpeg_path)
    except Exception as e:
        print(f"Failed to add_dll_directory: {e}")

print("Attempting to import torchcodec...")
try:
    import torchcodec
    from torchcodec.decoders import AudioDecoder
    print("SUCCESS: torchcodec imported and AudioDecoder found.")
except Exception as e:
    print(f"FAILURE: {e}")
    import traceback
    traceback.print_exc()

import torch 
print(f"PyTorch version: {torch.__version__}")
