import os
from multiprocessing import Pool
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--cuda_num', help='cuda num', required=True)
parser.add_argument('--topP', action="store_true", help='topP')
parser.add_argument('--start', type=int, help='start index')
parser.add_argument('--end', type=int, help='end index')
opt = parser.parse_args()

cuda = int(opt.cuda_num)

if opt.topP == True:
    toptype = "_topP"
else:
    toptype = ""

# g = os.walk("training_data/c-rnn-gan-master/data/classical")
# midi = []
# for path,d,filelist in g:
#     for filename in filelist:
#         if filename.endswith('mid'):
#             print("==========="*5)
#             print(path)
#             print(filename)
#             print("==========="*5)
#             midi.append(["/".join(path.split("/")[1:]), filename])

for index in tqdm(range(opt.start, opt.end)):
    os.system("make train_pickle{} CUDA={} NAME=p TYPE=xl N=400 DIR=JSB-Chorales-dataset FILE=jsb-chorales-16th.pkl INDEX={}".format(toptype, cuda, index))
    print("*******************************************")
    print("**************   END   ********************")
    print("*******************************************")