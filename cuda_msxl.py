import os
from multiprocessing import Pool
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--cuda_num', help='cuda num', required=True)
parser.add_argument('--start', type=int, help='start index')
parser.add_argument('--end', type=int, help='end index')
opt = parser.parse_args()
cuda = int(opt.cuda_num)

g = os.walk("training_data/midi")
midi = []
for path,d,filelist in g:
    for filename in filelist:
        if filename.endswith('mid'):
            print("==========="*5)
            print(path)
            print(filename)
            print("==========="*5)
            midi.append(["/".join(path.split("/")[1:]), filename])

print(len(midi))

subset = midi[opt.start:opt.end]
for path, mid in tqdm(subset):
    os.system("make train CUDA={} NAME=default TYPE=xl N=400 DIR={} FILE={}".format(cuda, path, mid))
    print("*******************************************")
    print("**************   END   ********************")
    print("*******************************************")