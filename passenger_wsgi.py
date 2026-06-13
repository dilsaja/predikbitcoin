import os
import sys

INTERP = os.path.expanduser("~/virtualenv/bitcoin_app/3.10/bin/python")

if sys.executable != INTERP:
    os.execl(INTERP, INTERP, *sys.argv)

sys.path.append(os.getcwd())

from app import application