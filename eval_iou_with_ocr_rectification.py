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

with open('ocr_info.pk','rb') as f:
    ocr_info = pickle.load(f)

with open('stopwords.txt',encoding='utf-8') as f:
    st = set(f.readlines())

findnum = re.compile(r'\d+')
zro = np.zeros((300,),dtype=np.float32)
findt = re.compile(r'"(.*?)"')
findk = re.compile(r'\[(.*?)\]')
findn = re.compile(r'[^\w\s]')
finde = re.compile(r'[A-Za-z,.:?!]')
osd = [args.input]

pat = re.compile(r'<answer>(.*?)</answer>')

fw = open('eval_iou_with_ocr_result.txt','w')

def bbox_diou(bboxes1, bboxes2):
    left = np.maximum(bboxes1[:, None, 0], bboxes2[:, 0])  # [N, M]
    top = np.maximum(bboxes1[:, None, 1], bboxes2[:, 1])
    right = np.minimum(bboxes1[:, None, 2], bboxes2[:, 2])
    bottom = np.minimum(bboxes1[:, None, 3], bboxes2[:, 3])
    inter_area = np.maximum(0, right - left) * np.maximum(0, bottom - top)
    area1 = (bboxes1[:, 2] - bboxes1[:, 0]) * (bboxes1[:, 3] - bboxes1[:, 1])  # [N,]
    area2 = (bboxes2[:, 2] - bboxes2[:, 0]) * (bboxes2[:, 3] - bboxes2[:, 1])  # [M,]
    union_area = area1[:, None] + area2 - inter_area
    iou = inter_area / (union_area + 1e-8)
    center_x1 = (bboxes1[:, 0] + bboxes1[:, 2]) / 2
    center_y1 = (bboxes1[:, 1] + bboxes1[:, 3]) / 2
    center_x2 = (bboxes2[:, 0] + bboxes2[:, 2]) / 2
    center_y2 = (bboxes2[:, 1] + bboxes2[:, 3]) / 2
    dist_center = (center_x1[:, None] - center_x2) ** 2 + (center_y1[:, None] - center_y2) ** 2
    left_enclose = np.minimum(bboxes1[:, None, 0], bboxes2[:, 0])
    top_enclose = np.minimum(bboxes1[:, None, 1], bboxes2[:, 1])
    right_enclose = np.maximum(bboxes1[:, None, 2], bboxes2[:, 2])
    bottom_enclose = np.maximum(bboxes1[:, None, 3], bboxes2[:, 3])
    dist_enclose = (right_enclose - left_enclose) ** 2 + (bottom_enclose - top_enclose) ** 2
    diou = iou - dist_center / (dist_enclose + 1e-8)
    return diou


def cal_iou(box1, box2):
    x_inter_min = np.maximum(box1[0], box2[0])
    y_inter_min = np.maximum(box1[1], box2[1])
    x_inter_max = np.minimum(box1[2], box2[2])
    y_inter_max = np.minimum(box1[3], box2[3])
    intersection_area = np.maximum(x_inter_max - x_inter_min, 0) * np.maximum(y_inter_max - y_inter_min, 0)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = area1 + area2 - intersection_area
    iou = intersection_area / union_area
    return iou

a=0
for d in osd:
    alls = []
    with open(d) as f:
        fl = f.read().splitlines()
        alls.extend([json.loads(l) for l in fl])
    sims = 0.0
    nums = 0.0
    if d.endswith('_cis') and (len(alls)!=11240):
        continue
    elif d.endswith('_cl') and (len(alls)!=9757):
        continue
    if d.endswith('_ctm') and (len(alls)!=4606):
        continue
    elif d.endswith('_test') and (len(alls)!=13949):
        continue
    for fjs in alls: 
      path = fjs['images'][0]['path']
      path2 = path.replace('/hy58/chenfan/','')
      thistext, thisbox, flag = ocr_info[path2]
      if not (fjs['objects'] is None):
        if True:
            nums = (nums + 1)
            pred = pat.search(fjs['response'].replace('\n',' '))
            if pred is None:
                pred = fjs['response']
            else:
                pred = pred.group(1).strip().replace('\n',' ')
            gt = pat.search(fjs['labels'].replace('\n',' ')).group(1).strip().replace('\n',' ')
            gt_num = [int(x) for x in fjs['objects']['bbox'][0]]
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
            if flag==1:
                predtext = pat.search(fjs['response'].replace('\n',' '))
                if predtext is None:
                    predtext = (fjs['response'].replace('\n',' '))
                else:
                    predtext = predtext.group(1).strip().replace('\n',' ')
                if True:#len(gtext)!=0:
                    ptext = findt.findall(predtext)
                    if len(ptext)==2:
                        pline = ptext[1]
                        ptext = ptext[0]
                        ptext = findn.sub('',ptext).strip() if len(ptext)!=0 else []
                        if not isinstance(ptext, list):
                            ocr_box = [thisbox[ni] for ni,n in enumerate(thistext) if (ptext in n)]
                            ocr_line = [findn.sub('',thistext[ni]).strip() for ni,n in enumerate(thistext) if ((ptext in n) and (len(thistext[ni])!=0))]
                            if len(ocr_line)>0:
                              if True:
                                pline_edits = np.array([edit_distance(pline, ocrl)/(max(len(ptext), len(ocrl))+1e-8) for ocrl in ocr_line])
                                if pline_edits.min()>0.2:
                                    # no match
                                    sims = (sims + cal_iou(pred_num, gt_num))
                                    continue
                                # line match
                                line_idxs = (pline_edits==pline_edits.min())
                                line_chars = ''.join([ocr_line[iii] for iii in range(len(ocr_line)) if line_idxs[iii]])
                                line_boxes = np.concatenate([ocr_box[iii] for iii in range(len(ocr_box)) if line_idxs[iii]])
                                box_use = [line_boxes[ni] for ni,n in enumerate(line_chars) if (n==ptext)]
                                if len(box_use)!=0:
                                    box_use = np.stack(box_use)
                                    if len(box_use)==1:
                                        pred_num = box_use[0].tolist()
                                    elif len(box_use)>1:
                                        pred_num = np.array([pred_num])
                                        dist = bbox_diou(np.array(pred_num), box_use).squeeze(0)
                                        pred_num = box_use[dist.argmin()].tolist()
                                    sims = (sims + cal_iou(pred_num, gt_num))
                                    continue
                              else:
                                  sims = (sims + cal_iou(pred_num, gt_num))
                                  continue
                        else:
                            sims = (sims + cal_iou(pred_num, gt_num))
                            continue
            elif flag==2:
                predtext = pat.search(fjs['response'].replace('\n',' '))
                if predtext is None:
                    predtext = (fjs['response'].replace('\n',' '))
                else:
                    predtext = predtext.group(1).strip().replace('\n',' ')
                ptext = findt.findall(predtext)
                if len(ptext)==2:
                    pline = ptext[1]
                    ptext = ptext[0]
                    ptext = findn.sub('',ptext).strip() if len(ptext)!=0 else []
                ptext_edits = np.array([edit_distance(pline, ocrl)/(max(len(ptext), len(ocrl))+1e-8) for ocrl in thistext])
                if ptext_edits.min()>0.2:
                    # no match
                    sims = (sims + cal_iou(pred_num, gt_num))
                    continue
                # match
                line_idxs = (ptext_edits==ptext_edits.min())
                line_texts = [thistext[iii] for iii in range(len(thistext)) if line_idxs[iii]]
                box_use = np.stack([thisbox[iii] for iii in range(len(thisbox)) if line_idxs[iii]])
                if len(box_use)==0:
                    sims = (sims + cal_iou(pred_num, gt_num))
                    continue
                elif len(box_use)==1:
                    pred_num = box_use[0].tolist()
                    sims = (sims + cal_iou(pred_num, gt_num))
                    continue
                elif len(box_use)>1:
                    pred_num = np.array([pred_num])
                    dist = bbox_diou(np.array(pred_num), box_use).squeeze(0)
                    pred_num = box_use[dist.argmin()].tolist()
                    sims = (sims + cal_iou(pred_num, gt_num))
                    continue
            try:
                sims = (sims + cal_iou(pred_num, gt_num))
            except:
                continue
        else:
            continue
    print("Json File:", d, "Number of Samples:", len(alls), "IoU:", sims/nums)
    fw.write('Json File: %s Number of Samples: %s IoU: %s\n'%(d, len(alls), sims/nums))
fw.close()
