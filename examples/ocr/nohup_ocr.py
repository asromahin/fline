import os
import sys

os.system("nohup sh -c '" + sys.executable + f" ocr_segmentation.py > log.txt" + "' &")
