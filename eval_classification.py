import os
import re
import io
import math
import json
import pickle
import base64
import requests
import argparse
import numpy as np
from tqdm import tqdm
from Levenshtein import distance as edit_distance

parser = argparse.ArgumentParser()
parser.add_argument("--input", required=True, type=str, help="Input inference result (json-file) for IoU evaluation.")
args = parser.parse_args()

with open('stopwords.txt',encoding='utf-8') as f:
    st = set(f.readlines())

zro = np.zeros((300,),dtype=np.float32)
findt = re.compile(r'"(.*?)"')
findn = re.compile(r'[^\w\s]')

osd = [args.input]

pat = re.compile(r'<answer>(.*?)</answer>')
fw = open('eval_realfake.txt','w')
for d in osd:
    alls = []
    with open(d) as f:
        fl = f.read().splitlines()
        alls.extend([json.loads(l) for l in fl])
    sims = 0.0
    for fjs in alls:
      try:
        pred = pat.search(fjs['response'].replace('\n',' '))
        if pred is None:
            pred = fjs['response'].replace('\n',' ')
        else:
            pred = pred.group(1).strip().replace('\n',' ')
        gt = pat.search(fjs['labels'].replace('\n',' ')).group(1).strip().replace('\n',' ')
        p1,p2 = pred.split('.')[:2]#.strip()
        g1,g2 = gt.split('.')[:2]#.strip()
        edits = edit_distance(p1.strip(), g1.strip())
        if edits==0:
            sims = (sims+1)
      except:
        print('error')
        continue
    print('accuracy', sims/len(alls))
    fw.write('Input: %s Accuracy: %s\n'%(d, sims/len(alls)))

fw.close()
