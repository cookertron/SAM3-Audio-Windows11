
import sys
import os
print("CWD:", os.getcwd())
print("Sys Path:")
for p in sys.path:
    print(p)
try:
    import dataset
    print("Dataset imported:", dataset)
except ImportError as e:
    print("Import failed:", e)
