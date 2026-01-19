import os
import re
from tqdm import tqdm
import requests
import json
import base64
from Levenshtein import distance as edit_distance
import re
import io
import math
import numpy as np
import argparse

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

fw = open('eval_ocr.txt', 'w')

for d in osd:
    alls = []
    with open(d) as f:
        fl = f.read().splitlines()
        alls.extend([json.loads(l) for l in fl])
    sims = 0.0
    nums = 0.0
    for fjs in alls:
      if True:
        pred = pat.search(fjs['response'].replace('\n',' '))
        if not (pred is None):
            pred = pred.group(1).strip().replace('\n',' ')
        else:
            pred = (fjs['response'].replace('\n',' '))
        gt = pat.search(fjs['labels'].replace('\n',' ')).group(1).strip().replace('\n',' ')
        gtext = findt.findall(gt)#[0]
        if len(gtext)!=0:
            gtext = gtext[0]
            nums = (nums + 1)
            ptext = findt.findall(pred)
            if len(ptext)!=0:
                ptext = ptext[0]
                #gtext = findn.sub('',gtext) if len(ptext)!=0 else []
                #ptext = findn.sub('',ptext) if len(ptext)!=0 else []
                #print(ptext, gtext, len(gtext))
                if len(gtext)!=0:
                    edits = (1-edit_distance(ptext, gtext)/max(len(ptext), len(gtext)))
                    sims =(sims + edits)
        # p1,p2 = pred.split('.')[:2]
        # g1,g2 = gt.split('.')[:2]
      else:
          continue
    print('Input:', d, 'OCR Accuracy:', ims/nums)
    fw.write('Input: %s OCR Accuracy: %s\n'%(d, sims/nums))

fw.close()
