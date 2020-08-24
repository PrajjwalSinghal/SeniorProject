import shutil
import os
from string import ascii_uppercase
try:
    shutil.rmtree('data')
except:
    pass

os.mkdir('data')
os.mkdir('data/train')
os.mkdir('data/val')
list  = ascii_uppercase[0:14]
for c in list:
    os.mkdir('data/train/'+c)
    os.mkdir('data/val/'+c)
