import os
import sys

os.system("nohup sh -c '" + sys.executable + f" gwd_box_verify.py > log.txt" + "' &")
