import os
from multiprocessing import Pool
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--cuda_num', help='cuda num', required=True)
parser.add_argument('--model', help='model type', required=True)
opt = parser.parse_args()

cuda = int(opt.cuda_num)

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

for index in tqdm(range(0, 382)):
    if opt.model == "ms":
        os.system("make test_pickle_topP CUDA={} NAME=p TYPE=xl N=400 INDEX={} DIR=JSB-Chorales-dataset FILE=jsb-chorales-16th.pkl ".format(cuda, index))
    elif opt.model == "ss":
        os.system("make test_pickle_single_topP CUDA={} NAME=sp TYPE=xl N=400 INDEX={} DIR=JSB-Chorales-dataset FILE=jsb-chorales-16th.pkl".format(cuda, index))
    else:
        print("error model type")
        break
    print("*******************************************")
    print("**************   END   ********************")
    print("*******************************************")