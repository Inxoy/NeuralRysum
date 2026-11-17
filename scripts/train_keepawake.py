"""Run a training module while asking Windows not to idle-sleep.

    python scripts/train_keepawake.py --data data\\organ --out runs\\rysum
    python scripts/train_keepawake.py -m neuralrysum.train_pitch --out runs\\pitch

Without ``-m`` it runs ``neuralrysum.train``; all other arguments go to the
module. Arguments are required, so a bare call cannot start a run with
default paths by accident. The request ends with the process and changes no
power settings. The display may still turn off; closing a laptop lid or
choosing Sleep still sleeps.
"""
import ctypes
import importlib
import sys

ES_CONTINUOUS = 0x80000000
ES_SYSTEM_REQUIRED = 0x00000001

if __name__ == "__main__":
    args = sys.argv[1:]
    module = "neuralrysum.train"
    if args[:1] == ["-m"]:
        module, args = args[1], args[2:]
    if "--out" not in args:
        sys.exit("refusing to start without --out")
    sys.argv = [module] + args
    if sys.platform == "win32":
        ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)
    importlib.import_module(module).main()
