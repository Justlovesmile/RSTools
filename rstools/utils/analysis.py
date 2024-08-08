import os
import json
from rstools.utils.file import get_file_from_this_dir

def analysis_rslabel(jsondir):
    cls_dict = {}
    jsonpaths = get_file_from_this_dir(jsondir, ext='json')
    for jsonpath in jsonpaths:
        with open(jsonpath, 'r') as f:
            content = json.load(f)
        for shape in content['shapes']:
            label = shape['label']
            if label not in cls_dict.keys():
                cls_dict[label] = 1
            else:
                cls_dict[label] += 1
    print_analysis(cls_dict)
    return cls_dict


def analysis_otherjson(jsondir):
    cls_dict = {}
    jsonpaths = get_file_from_this_dir(jsondir, ext='json')
    for jsonpath in jsonpaths:
        with open(jsonpath, 'r') as f:
            content = json.load(f)
        for item in content['markResult']['features']:
            label = item['title']
            if label not in cls_dict.keys():
                cls_dict[label] = 1
            else:
                cls_dict[label] += 1
    print_analysis(cls_dict)
    return cls_dict


def analysis_dota(txtdir):
    cls_dict = {}
    txtpaths = get_file_from_this_dir(txtdir, ext='txt')
    for txtpath in os.listdir(txtpaths):
        with open(txtpath,'r') as f:
            lines = f.readlines()
        for line in lines:
            splitline = line.strip().split()
            if len(splitline) < 9:
                print(f"[WARNING] Skip {txtpath}: {line}")
                continue
            label = splitline[8]
            if label not in cls_dict.keys():
                cls_dict[label] = 1
            else:
                cls_dict[label] += 1
    print_analysis(cls_dict)
    return cls_dict


def print_analysis(cls_dict):
    total_num = 0
    print("="*15)
    names = list(cls_dict.keys())
    print(names)
    print("="*15)
    for idx,name in enumerate(sorted(names)):
        print(f"[{idx+1}]\t[Found {cls_dict[name]}]\t{name}")
        total_num += cls_dict[name]
    print(f"[Total]\t[Category: {len(names)}]\t[Targets: {total_num}]")