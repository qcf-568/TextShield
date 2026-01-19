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
from rouge import Rouge
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--input", required=True, type=str, help="Input inference result (json-file) for IoU evaluation.")
args = parser.parse_args()

rouges = Rouge()

with open('stopwords.txt',encoding='utf-8') as f:
    st = set(f.readlines())

zro = np.zeros((300,),dtype=np.float32)
findt = re.compile(r'"(.*?)"')
findn = re.compile(r'[^\w\s]')

osd = [args.input]

pat = re.compile(r'<answer>(.*?)</answer>')
pat2 = re.compile(r'<think>(.*?)</think>')

def cosine_similarity(v1, v2):
    # 计算两个向量的点积
    dot_product = np.dot(v1, v2)
    # 计算两个向量的模
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    # 计算余弦相似度
    similarity = dot_product / (norm_v1 * norm_v2)
    return similarity

def load_vectors(fname):
    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    n, d = map(int, fin.readline().split())
    data = {}
    for line in tqdm(fin):
        tokens = line.rstrip().split(' ')
        data[tokens[0]] = np.array(tokens[1:])
    return data

model = load_vectors('wiki-news-300d-1M.vec')

zro = np.zeros((300,),dtype=np.float32)

frec = open('eval_reasoning.txt','w')
for d in osd:
    alls = []
    with open(d) as f:
        fl = f.read().splitlines()
        alls.extend([json.loads(l) for l in fl])
    cos = 0.0
    bleu = 0.0
    rouge = 0.0
    nums = 0.0
    for fjs in alls:
      try:
        pred = pat.search(fjs['response'].replace('\n',' '))#.group(1).strip().replace('\n',' ')
        if pred is None:
            pred = fjs['response'].replace('\n',' ')
        else:
            pred = pred.group(1).strip().replace('\n',' ')
        gt = pat.search(fjs['labels'].replace('\n',' ')).group(1).strip().replace('\n',' ')
        p1,p2 = pred.split('.')[:2]
        g1,g2 = gt.split('.')[:2]
        if ('is real' in g1):
            continue
        else:
            nums = (nums + 1)
            edits = edit_distance(p1.strip()+'1', g1.strip()+'1')
            if edits==0:
                    pred = pat2.search(fjs['response'].replace('\n',' ')).group(1).strip().replace('\n',' ')
                    gt = pat2.search(fjs['labels'].replace('\n',' ')).group(1).strip().replace('\n',' ')  
                    ptext = findt.findall(pred)
                    for p in ptext:
                        pred = pred.replace(p, '')
                    pred = findn.sub('', pred)
                    ptext = findn.sub('',ptext[0]) if len(ptext)!=0 else []
                    gtext = findt.findall(gt)
                    for p in gtext:
                        gt = gt.replace(p, '')
                    gt = findn.sub('', gt)
                    gtext = findn.sub('',gtext[0]) if len(gtext)!=0 else []
                    predcos = np.array([model.get(x,zro) for x in pred.split(' ') if ((len(x)!=0) and (not (x in st)))], dtype=np.float32)
                    gtcos = np.array([model.get(x,zro) for x in gt.split(' ') if ((len(x)!=0) and (not (x in st)))], dtype=np.float32)
                    predcos = predcos.mean(0)
                    gtcos = gtcos.mean(0)
                    cos_ = cosine_similarity(predcos, gtcos)
                    rouge_ = rouges.get_scores(pred, gt, avg=True)['rouge-l']['f']
                    bleu_ = sentence_bleu([gt.split()], pred.split())
                    if cos_<2:
                        cos = (cos + cos_)
                    if rouge_<2:
                        rouge = (rouge + rouge_)
                    if bleu_<2:
                        bleu = (bleu + bleu_)

      except:
        print('error')
        nums = (nums +1)
        continue
    coss = cos/nums
    rougess = rouge/nums
    bleus = bleu/nums
    print('Reasoning Score:', (coss+bleus+rougess)/3.0)
    frec.write('Input: %s'%d+'Reasoning Score: '+str((coss+bleus+rougess)/3.0))
frec.close()
