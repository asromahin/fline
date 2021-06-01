import os
import sys

os.system("nohup sh -c '" + sys.executable + f" train_obj_detection.py > log.txt" + "' &")
