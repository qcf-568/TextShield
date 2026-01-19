import os
import json
import pickle
import random
import argparse
import imagesize
import numpy as np
from tqdm import tqdm
from Levenshtein import distance as edit_distance

parser = argparse.ArgumentParser()
parser.add_argument("--input", required=True, type=str, help="Input image dir.")
args = parser.parse_args()

osd = [os.path.join(args.input, x) for x in os.listdir(args.input)]
jsf = open('%s.jsonl'%args.input.rstrip('/'), 'w')

for f in tqdm(osd):
    w, h = imagesize.get(f)
    assert h%28==0, 'image height must be times of 28'
    assert w%28==0, 'image height must be times of 28'
    itm = {"messages": [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": """<image> Is this image real, entirely generated, or tampered? If it has been tampered, what method was used, and what are the content and bounding box coordinates of the tampered text? Output the thinking process in <think> </think> and \n final answer (number) in <answer> </answer> tags. \n Here is an example answer for a real image: <answer> This image is real. </answer> Here is an example answer for an entirely generated image: <answer> This image is entirely generated. </answer> Here is an example answer for a locally tampered image: <answer> This image is tampered. It was tampered by copy-paste. The tampered text reads "small" in the text line "a small yellow flower", and it is located at ... </answer>"""}]}
    jsf.write(json.dumps(itm, ensure_ascii=False)+'\n')

jsf.close()
