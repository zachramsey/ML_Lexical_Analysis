import traceback as tb
import sys
import os

"""
@brief  Print traceback of error and exit program
"""
def print_tb():
    print("--------------------------------")
    stack = tb.extract_tb(sys.exc_info()[2])
    for frame in stack:
        print(f"{os.path.basename(frame.filename)}:{frame.lineno} | {frame.line}")
    print(f"\nError: {sys.exc_info()[1]}")
    print("--------------------------------")
    sys.exit(0)