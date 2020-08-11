import numpy as np
import os
import torch
import json
import glob
import cv2
import shutil
import tqdm
from collections import defaultdict
import pandas as pd
import openpyxl
import copy

def get_full_info_dict(file_path):
    full_info_dict = None
    if file_path[:8] == "sub_dirs":
        return full_info_dict
    if file_path.split('.')[-1] == 'txt':
        full_info_dict = full_info.build_full_info_dict(file_path, is_thick=False)
    elif file_path.split('.')[-1] == 'json':
        with open(file_path) as f:
            full_info_dict = json.load(f)
    elif file_path.split('.')[-1] == 'csv':
        df = pd.read_csv(file_path)
        df = df.fillna('0')
        full_info_dict = csv2dict(df)
    else:
        print("Unsupport type {}".format(file_path))
    return full_info_dict

def combin_full_info_dict(file_path):
    if os.path.isfile(file_path):
        return get_full_info_dict(file_path)
    full_info_dict_combin = {}
    for dirpath, dirnames, filenames in os.walk(file_path, followlinks=True):
        filenames = [f for f in filenames if ".txt" in f or ".json" in f or ".csv"]
        if len(filenames) == 0:
            continue
        for full_info_name in filenames:
            full_info_path = os.path.join(dirpath, full_info_name)
            full_info_dict = get_full_info_dict(full_info_path)
            full_info_dict_combin.update(full_info_dict)
    return full_info_dict_combin

def load_string_list(file_path, is_utf8=False):
    '''
    Load string list from mitok file
    '''
    try:
        if is_utf8:
            f = codecs.open(file_path, 'r', 'utf-8')
        else:
            f = open(file_path)
        l = []
        for item in f:
            item = item.strip()
            if len(item) == 0:
                continue
            l.append(item)
        f.close()
    except IOError:
        print('open error', file_path)
        return None
    else:
        return l

def save_string_list(file_path, l, is_utf8=False, save_pattern = 'w'):
    """
    Save string list as mitok file
    - file_path: file path
    - l: list store strings
    """
    if is_utf8:
        f = codecs.open(file_path, 'w', 'utf-8')
    else:
        f = open(file_path, save_pattern)
    for item in l[:-1]:
        f.write(item + '\n')
    if len(l) >= 1:
        f.write(l[-1])
    f.close()
    
def update_dict_list(dict_list, subdir_full_info, cls_list, num_pngs=1):
    for idx, cls in enumerate(cls_list):
        key = subdir_full_info[cls]
        if cls == "kernel":
            key = get_kernel_cls(key)
        elif cls == "manufacturer":
            key = get_manufacturer(key)
        if len(dict_list) < idx*2+1:
            _dict_ct = defaultdict(int)
            _dict_slice = defaultdict(int)
            _dict_ct[key]+=1
            _dict_slice[key]+=1
            dict_list.append(_dict_ct)
            dict_list.append(_dict_slice)
        else:
            dict_list[idx*2][key] += 1
            dict_list[idx*2+1][key]+= num_pngs
    return dict_list

def add_to_whole_dict(cls_list, whole_dict, dict_list):
    assert 2 * len(cls_list) == len(dict_list), "Class list not match dict list"
    for idx, cls in enumerate(cls_list):
        whole_dict[cls] = {cls+"_ct":dict_list[idx*2], cls+"_slice":dict_list[idx*2+1]}
    return whole_dict

def analy_full_info_by_subdirlist(full_info_dict, subdir_list, ignore_list=[]):
    flip_dict = defaultdict(int)
    whole_dict = {}
    cls_list = ["kernel", "slice_thickness", "manufacturer"]
    dict_list = []
    kernel_dict_ct = defaultdict(int)
    kernel_dict_slice = defaultdict(int)
    
    for subdir in subdir_list:
        num_ct = 0
        # import pdb; pdb.set_trace()
        if subdir not in full_info_dict.keys():
            print(subdir)
            continue
        dict_list = update_dict_list(dict_list, full_info_dict[subdir], cls_list, 1)
        num_ct+=1
    print(num_ct)
    # print(kernel_dict, nums)
    whole_dict = add_to_whole_dict(cls_list, whole_dict, dict_list)
    return whole_dict

def get_subdir_list_from_path(root_path):
    std_subdir_list = []
    for dirpath, dirnames, filenames in os.walk(root_path):
        if len(filenames) > 5:
            subdir = '/'.join(dirpath.split('/')[-3:])
            if subdir in std_subdir_list:
                print(subdir)
            else:
                std_subdir_list.append(subdir)
    return std_subdir_list


def write_excel(dict_file, save_path="", save_name="result.xlsx"):
    if type(dict_file) == dict:
        dict_file = dict_file
    else:
        print("Unsupport type {}".format(type(dict_file)))
    if save_path == "":
        return
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        
    save_file_path = os.path.join(save_path, save_name)
    for key in dict_file.keys():
        df = pd.DataFrame(dict_file[key])
        if not os.path.isfile(save_file_path):
            df.to_excel(save_file_path, sheet_name=key)
        else:
            excel_writer = pd.ExcelWriter(save_file_path, engine="openpyxl")
            add_sheet(df, excel_writer, key)

def add_sheet(data, excel_writer, sheet_name):
    book = openpyxl.load_workbook(excel_writer.path)
    if sheet_name in book.sheetnames:
        return
    excel_writer.book = book
    data.to_excel(excel_writer=excel_writer, sheet_name=sheet_name, index=True)
 
    excel_writer.close()
    
def updata_dict(left_dict, right_dict):
    assert left_dict.keys() == right_dict.keys()
    for key in left_dict.keys():
        pass
    
def get_kernel_cls(kernel):
    kernel = kernel.strip().lower()
    if len(kernel) > 1 and (kernel[0] == 'b' or kernel[0] == 'i' or kernel[0] == 'h') and (kernel[1] >= '0' and kernel[1] <= '9'):
        num = str2int(kernel[1:])
        if(num <= 40):
            return "STAN"
        elif(num <= 60):
            return "LUNG"
        else:
            return "BONE"
    
    if len(kernel) > 2 and (kernel[:2] == "br" or kernel[:2] == "ir" or kernel[:2] == "hr" or kernel[:2] == "bl") and (kernel[2] >= '0' and kernel[2] <= '9'):
        num = str2int(kernel[2:])
        if(num <= 40):
            return "STAN"
        elif(num <= 60):
            return "LUNG"
        else:
            return "BONE"
    
    LUNG = ["Y-Sharp", "Lung Enhanced", "YC", "L", "B_SHARP_C", "Lung", "B_VSHARP_C", "FC50", "FC51", "FC52", "FC56",
            "FC53", "FC83", "FC84", "FC85", "FC86"]
    BONE = ["Bone", "B_VSHARP_B", "FC30", "FC31", "FC81", "YB"]
    STAN = ["STANDARD", "B","B_SOFT_B", "B_SOFT_C", "FC01", "FC02", "FC03", "FC04", "FC05", "FC08", "FC09", "FC13",
            "FC14", "FC18", "FC18-H"]
    
    kernel_dict = {"LUNG":LUNG, "BONE":BONE, "STAN":STAN}
    for key in kernel_dict.keys():
        kernel_list = kernel_dict[key]
        for kernel_ in kernel_list:
            kernel_ = kernel_.strip().lower()
            if kernel == kernel_ or kernel == kernel_+'s' or kernel == kernel_+'f':
                return key
    return "other"

def str2int(str):
    num = 0
    for i in str:
        if i >= '0' and i <= '9':
            num = num*10+int(i)
        else:
            break
    return num

def get_manufacturer(manufacturer):
    manufacturer = manufacturer.strip().lower()
    manufacturer_list = ["GE MEDICAL SYSTEMS", "Philips", "SIEMENS", "TOSHIBA", "UIH"]
    for manufacturer_ in manufacturer_list:
        manufacturer_ = manufacturer_.lower().strip()
        if manufacturer in manufacturer_:
            return manufacturer_
    return None

# def csv2dict(df):
#     values = df.values
#     fullinfo_dict = {}
#     columes = df.columns[2:]
#     for idx in df.index:
#         val = values[idx][1:]
#         if val[1] != '[Valid]':
#             continue
#         fullinfo_dict[val[0]] = {}
#         for i, col in enumerate(columes):
#             fullinfo_dict[val[0]][col] = val[i+1]
#     return fullinfo_dict
def csv2dict(df):
    fullinfo_dict = {}
    for i, rows in df.iterrows():
        if rows["is_valid"] != '[Valid]':
            continue
        fullinfo_dict[rows["sub_dir"]] = {}
        for key in rows.keys():
            fullinfo_dict[rows["sub_dir"]][key] = rows[key]
    return fullinfo_dict
