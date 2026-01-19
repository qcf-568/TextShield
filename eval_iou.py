import os
import re
import imagesize
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

findk = re.compile(r'\[(.*?)\]')
findnum = re.compile(r'\d+')
zro = np.zeros((300,),dtype=np.float32)
findt = re.compile(r'"(.*?)"')
findn = re.compile(r'[^\w\s]')

osd = [args.input]

pat = re.compile(r'<answer>(.*?)</answer>')

fw = open('eval_iou.txt','w')

def cal_iou(box1, box2):
    # box1: [x_min, y_min, x_max, y_max]
    # box2: [x_min, y_min, x_max, y_max]

    # 计算交集区域的坐标
    # print(box1[0], box2[0])
    x_inter_min = np.maximum(box1[0], box2[0])
    y_inter_min = np.maximum(box1[1], box2[1])
    x_inter_max = np.minimum(box1[2], box2[2])
    y_inter_max = np.minimum(box1[3], box2[3])

    # 计算交集面积
    intersection_area = np.maximum(x_inter_max - x_inter_min, 0) * np.maximum(y_inter_max - y_inter_min, 0)

    # 计算两个框的面积
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

    # 计算并集面积
    union_area = area1 + area2 - intersection_area

    # 计算IoU
    iou = intersection_area / union_area
    return iou


for d in osd:
    alls = []
    with open(d) as f:
        fl = f.read().splitlines()
        alls.extend([json.loads(l) for l in fl])
    sims = 0.0
    nums = 0.0
    for fjs in alls:
      if not (fjs['objects'] is None):
        if True:
            nums = (nums + 1)
            pred = pat.search(fjs['response'])
            if pred is None:
                pred = (fjs['response'])
            else:
                pred = pred.group(1).strip().replace('\n',' ')
            gt = pat.search(fjs['labels']).group(1).strip().replace('\n',' ')
            if 'tampered' in pred:
                gt_num = [int(x) for x in fjs['objects']['bbox'][0]]
                pred_num = [int(x) for x in findnum.findall(pred)]
                if 'box' in pred:
                    predb = pred.split('box')[-2]
                    pred_num = [int(x) for x in findnum.findall(predb)]
                elif ('[' in pred) and (']' in pred):
                    predb = ' '.join(findk.findall(pred))
                    pred_num = [int(x) for x in findnum.findall(predb)]
                else:
                    pred_num = [int(x) for x in findnum.findall(pred)]
                if len(pred_num)!=4:
                    continue
                try:
                    # print(d, w,h, pred_num, gt_num)
                    sims = (sims + cal_iou(pred_num, gt_num))
                except:
                    print('error')
                    continue
        else:
            print('error', fjs['response'])
            continue
    print('Input: ', d, 'IoU: ', sims/nums)
    fw.write('Input: %s IoU: %.3f\n'%(d, sims/nums))

fw.close()
