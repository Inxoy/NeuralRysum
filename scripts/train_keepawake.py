"""Run ``neuralrysum.train`` while asking Windows not to idle-sleep.

    python scripts/train_keepawake.py --data data\\organ --out runs\\rysum

Takes the same arguments as ``python -m neuralrysum.train``. The request ends
with the process and changes no power settings. The display may still turn
off; closing a laptop lid or choosing Sleep still sleeps.
"""
import ctypes
import sys

from neuralrysum.train import main

ES_CONTINUOUS = 0x80000000
ES_SYSTEM_REQUIRED = 0x00000001

if __name__ == "__main__":
    if sys.platform == "win32":
        ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)
    main()
