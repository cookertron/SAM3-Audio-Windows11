
import os
import ctypes
import sys

ffmpeg_path = r"E:\Software\ffmpeg\bin"
os.environ["PATH"] = ffmpeg_path + os.pathsep + os.environ["PATH"]

dll_name = "avcodec-60.dll"
dll_path = os.path.join(ffmpeg_path, dll_name)

print(f"Testing load of {dll_path}")
try:
    # Try loading with explicit path
    lib = ctypes.CDLL(dll_path)
    print("Successfully loaded via full path")
except Exception as e:
    print(f"Failed to load via full path: {e}")

try:
    # Try loading via search path
    lib = ctypes.CDLL(dll_name)
    print("Successfully loaded via name")
except Exception as e:
    print(f"Failed to load via name: {e}")
