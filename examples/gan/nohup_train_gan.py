import os
import sys

os.system("nohup sh -c '" + sys.executable + f" fake_gen_classifier.py > log.txt" + "' &")
