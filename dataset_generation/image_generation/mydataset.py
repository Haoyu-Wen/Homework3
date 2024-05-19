import json
import shutil
import os
from tqdm import tqdm
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
# import h5py
import torch

name='CLEVR'
name1 = name+"hat"

if not os.path.exists(f"{name}/TRAIN/caplens"):
    os.makedirs(f"{name}/TRAIN/caplens", exist_ok=True)
if not os.path.exists(f"{name}/TRAIN/captions"):
    os.makedirs(f"{name}/TRAIN/captions", exist_ok=True)

if not os.path.exists(f"{name}/TEST/caplens"):
    os.makedirs(f"{name}/TEST/caplens", exist_ok=True)
if not os.path.exists(f"{name}/TEST/captions"):
    os.makedirs(f"{name}/TEST/captions", exist_ok=True)

# if not os.path.exists("./CLEVR/imgs"):
    # os.mkdir("./CLEVR/imgs")

if not os.path.exists(f"{name}/TRAIN/imgs/0"): # before
    os.makedirs(f"{name}/TRAIN/imgs/0", exist_ok=True)
if not os.path.exists(f"{name}/TRAIN/imgs/1"): # during
    os.makedirs(f"{name}/TRAIN/imgs/1", exist_ok=True)
if not os.path.exists(f"{name}/TRAIN/imgs/2"): # after
    os.makedirs(f"{name}/TRAIN/imgs/2", exist_ok=True)

if not os.path.exists(f"{name}/TEST/imgs/0"): # before
    os.makedirs(f"{name}/TEST/imgs/0", exist_ok=True)
if not os.path.exists(f"{name}/TEST/imgs/1"): # during
    os.makedirs(f"{name}/TEST/imgs/1", exist_ok=True)
if not os.path.exists(f"{name}/TEST/imgs/2"): # after
    os.makedirs(f"{name}/TEST/imgs/2", exist_ok=True)
if not os.path.exists(f"{name}/wordmap"):
    os.mkdir(f"{name}/wordmap")

if not os.path.exists(f"{name1}/TRAIN/imgs/0"):
    os.makedirs(f"{name1}/TRAIN/imgs/0", exist_ok=True)
if not os.path.exists(f"{name1}/TRAIN/imgs/1"):
    os.makedirs(f"{name1}/TRAIN/imgs/1", exist_ok=True)
# if not os.path.exists(f"{name1}/TRAIN/imgs/2"):
#     os.makedirs(f"{name1}/TRAIN/imgs/2")
if not os.path.exists(f"{name1}/TEST/imgs/0"):
    os.makedirs(f"{name1}/TEST/imgs/0", exist_ok=True)
if not os.path.exists(f"{name1}/TEST/imgs/1"):
    os.makedirs(f"{name1}/TEST/imgs/1", exist_ok=True)
# if not os.path.exists(f"{name1}/TEST/imgs/2"):
#     os.makedirs(f"{name1}/TEST/imgs/2")
if not os.path.exists(f"{name1}/TRAIN/captions"):
    os.makedirs(f"{name1}/TRAIN/captions", exist_ok=True)
if not os.path.exists(f"{name1}/TRAIN/caplens"):
    os.makedirs(f"{name1}/TRAIN/caplens", exist_ok=True)
if not os.path.exists(f"{name1}/TEST/captions"):
    os.makedirs(f"{name1}/TEST/captions", exist_ok=True)
if not os.path.exists(f"{name1}/TEST/caplens"):
    os.makedirs(f"{name1}/TEST/caplens", exist_ok=True)
if not os.path.exists(f"{name1}/wordmap"):
    os.makedirs(f"{name1}/wordmap", exist_ok=True)



word_vac = ["<start>", "<end>", "<pad>", "<unk>", "<sep>"]
if os.path.exists("./CLEVR/wordmap/wordmap.json"):
    with open("./CLEVR/wordmap/wordmap.json", 'r') as f:
        word_vac = json.load(f)
change_caption_json_path = "change_caption.json"
change_caption_reverse_json_path = "change_caption_reverse.json"
data_dir = "../output"
padding_len = 82
padding_len2 = 120  


def get_caption_json(caption,state="<sep>", end_flag=0):
    caption_json = []
    caption_json.append(word_vac.index(state))
    for word in caption.split():
        if word not in word_vac:
            word_vac.append(word)
        caption_json.append(word_vac.index(word))
    if end_flag==1:
        caption_json.append(word_vac.index("<end>"))
    length = len(caption_json)
    while len(caption_json) < padding_len:
        caption_json.append(word_vac.index("<pad>"))
    return  caption_json, length

def reverse_caption(captions:list, caplens:list):
    word_vac_json = {k:v for v,k in enumerate(word_vac)}
    captions.reverse()
    caplens.reverse()
    captions[0].remove(word_vac_json["<sep>"])
    captions[0].insert(0, word_vac_json["<start>"])
    captions[0].remove(word_vac_json["<end>"])

    captions[-1].remove(word_vac_json["<start>"])
    captions[-1].insert(0, word_vac_json["<sep>"])
    captions[-1].insert(captions[-1].index(word_vac_json['<pad>']) ,word_vac_json['<end>'])

    caplens[0] = caplens[0]-1

    caplens[-1] = caplens[-1]+1
    return captions, caplens

# def generate_data():

    all_caption_json, all_caplen_json, all_caption_reverse_json, all_caplen_reverse_json = [], [], [], []
    with open(change_caption_json_path, 'r')as f:
        captions = json.load(f)
    with open(change_caption_reverse_json_path, 'r') as f:
        captions_reverse = json.load(f)
    
    train_num = int(0.9*len(captions))
    for idx, caption in enumerate(captions):    
        # if idx == 2:
        #     break    
        for change_idx, change_caption in enumerate(caption['change_captions']):
            end_flag = 0
            if change_idx==0:
                state="<start>"
            else:
                state="<sep>"
            if change_idx==len(caption['change_captions'])-1:
                end_flag=1
            t_caption, t_caplen = get_caption_json(change_caption, state, end_flag)
            all_caption_json.append(t_caption)
            all_caplen_json.append(t_caplen)
        # img 1
        source_path = os.path.join(os.path.join(data_dir, 'images'), caption['image_id'])
        if (idx+1)<=train_num:
            tgt_path = os.path.join("./CLEVR/TRAIN/imgs/0", str(idx)+".png")
        else:
            tgt_path = os.path.join("./CLEVR/TEST/imgs/0", str(idx-train_num)+".png")
        shutil.copy2(source_path, tgt_path)
        # img 2
        source_path = os.path.join(os.path.join(data_dir, 'images'), caption['image_id'].split(".png")[0]+"_change"+"2"+"".join(["_"+record for record_id, record in enumerate(caption['change_record']) if record_id<=2])+".png") 
        if (idx+1)<=train_num:
            tgt_path = os.path.join("./CLEVR/TRAIN/imgs/1", str(idx)+".png")
        else:
            tgt_path = os.path.join("./CLEVR/TEST/imgs/1", str(idx-train_num)+".png")
        shutil.copy2(source_path, tgt_path)
        # img 3
        source_path = os.path.join(os.path.join(data_dir, 'images'), caption['image_id'].split(".png")[0]+"_change"+"5"+"".join(["_"+record for record in caption['change_record']])+".png") 
        if (idx+1)<=train_num:
            tgt_path = os.path.join("./CLEVR/TRAIN/imgs/2", str(idx)+".png")
        else:
            tgt_path = os.path.join("./CLEVR/TEST/imgs/2", str(idx-train_num)+".png")
        # tgt_path = os.path.join("./CLEVR/imgs/2", str(idx)+".png")
        shutil.copy2(source_path, tgt_path)
    
    # reverse captions (don't copy file--save space...)
    for idx, caption in enumerate(captions_reverse):    
        # if idx == 2:
        #     break
        tmp_caption = []
        tmp_caplen = []    
        for change_idx, change_caption in enumerate(caption['change_captions']):
            end_flag = 0
            if change_idx==0:
                state="<start>"
            else:
                state="<sep>"
            if change_idx==len(caption['change_captions'])-1:
                end_flag=1
            t_caption, t_caplen = get_caption_json(change_caption,state, end_flag)          
            tmp_caption.append(t_caption)
            tmp_caplen.append(t_caplen)
        tmp_caplen, tmp_caplen = reverse_caption(tmp_caption, tmp_caplen)
        all_caption_reverse_json.extend(tmp_caption)
        all_caplen_reverse_json.extend(tmp_caplen)

    train_num = 6*train_num # captions per image = 6
    # train
    with open("CLEVR/TRAIN/caplens/caplens.json", 'w') as f:
        json.dump(all_caplen_json[:train_num], f)
    with open("CLEVR/TRAIN/captions/captions.json", 'w') as f:
        json.dump(all_caption_json[:train_num], f)
    with open("CLEVR/TRAIN/caplens/caplens_reverse.json", 'w') as f:
        json.dump(all_caplen_reverse_json[:train_num], f)
    with open("CLEVR/TRAIN/captions/captions_reverse.json", 'w') as f:
        json.dump(all_caption_reverse_json[:train_num], f)

    # test
    with open("CLEVR/TEST/caplens/caplens.json", 'w') as f:
        json.dump(all_caplen_json[train_num:], f)
    with open("CLEVR/TEST/captions/captions.json", 'w') as f:
        json.dump(all_caption_json[train_num:], f)
    with open("CLEVR/TEST/caplens/caplens_reverse.json", 'w') as f:
        json.dump(all_caplen_reverse_json[train_num:], f)
    with open("CLEVR/TEST/captions/captions_reverse.json", 'w') as f:
        json.dump(all_caption_reverse_json[train_num:], f)

    word_vac_json = {k:v for v,k in enumerate(word_vac)}
    with open("CLEVR/wordmap/wordmap.json", 'w') as f:
        json.dump(word_vac_json, f)
    
def get_caption_json_2(captions:list):
    caption_res, caplen_res = [], 0
    caption_res.append(word_vac.index("<start>"))

    for idx, caption in enumerate(captions):
        for word in caption.split():
            if word not in word_vac:
                word_vac.append(word)
            caption_res.append(word_vac.index(word))
        if idx!=len(captions)-1:
            caption_res.append(word_vac.index("<sep>"))
        else:
            caption_res.append(word_vac.index("<end>"))
    caplen_res = len(caption_res)
    while len(caption_res) < padding_len2:
        caption_res.append(word_vac.index("<pad>"))
    return caption_res, caplen_res
    
def get_caption_json_reverse(captions:list):
    captions_res, caplens_res = [], 0
    t_caption, tt_caption = [], []
    captions_res.append(word_vac.index("<start>"))
    for caption in captions:
        t_caption = []
        for word in caption.split():
            if word not in word_vac:
                word_vac.append(word)
            t_caption.append(word_vac.index(word))
        tt_caption.append(t_caption)
    tt_caption.reverse()
    for id, caption_json in enumerate(tt_caption):
        captions_res.extend(caption_json)
        if id!=len(tt_caption)-1:
            captions_res.append(word_vac.index("<sep>"))
        else:
            captions_res.append(word_vac.index("<end>"))
    caplens_res = len(captions_res)
    while len(captions_res) < padding_len2:
        captions_res.append(word_vac.index("<pad>"))
    return captions_res, caplens_res

# def get_caption_json_2imgs(caption:list):

def generate_data(name, changes):

    all_caption_json, all_caplen_json, all_caption_reverse_json, all_caplen_reverse_json = [], [], [], []
    with open(change_caption_json_path, 'r')as f:
        captions = json.load(f)
    with open(change_caption_reverse_json_path, 'r') as f:
        captions_reverse = json.load(f)
    
    train_num = int(0.9*len(captions))
    # past
    # for idx, caption in enumerate(captions):    
    #     # past, get captions like [[], [], [] pair 1 ---[], [], []...]
    #     for change_idx, change_caption in enumerate(caption['change_captions']):
    #         end_flag = 0
    #         if change_idx==0:
    #             state="<start>"
    #         else:
    #             state="<sep>"
    #         if change_idx==len(caption['change_captions'])-1:
    #             end_flag=1
    #         t_caption, t_caplen = get_caption_json(change_caption, state, end_flag)
    #         all_caption_json.append(t_caption)
    #         all_caplen_json.append(t_caplen)
        
    for idx, caption in enumerate(captions): 
        # now, get captions like [[]pair 1, []pair 2...]
        cap, caplen = get_caption_json_2(caption['change_captions'])
        all_caption_json.append(cap)
        all_caplen_json.append(caplen)

        source_path = os.path.join(os.path.join(data_dir, 'images'), caption['image_id'])
        if (idx+1)<=train_num:
            tgt_path = os.path.join(f"./{name}/TRAIN/imgs/0", str(idx)+".png")
        else:
            tgt_path = os.path.join(f"./{name}/TEST/imgs/0", str(idx-train_num)+".png")
        # img 1
        shutil.copy2(source_path, tgt_path)
        # img 2
        source_path = os.path.join(os.path.join(data_dir, 'images'), caption['image_id'].split(".png")[0]+"_change"+str(changes-1)+"".join(["_"+record for record_id, record in enumerate(caption['change_record']) if record_id<=(changes-1)])+".png") 
        if (idx+1)<=train_num:
            tgt_path = os.path.join(f"./{name}/TRAIN/imgs/1", str(idx)+".png")
        else:
            tgt_path = os.path.join(f"./{name}/TEST/imgs/1", str(idx-train_num)+".png")
        shutil.copy2(source_path, tgt_path)
        # img 3
        source_path = os.path.join(os.path.join(data_dir, 'images'), caption['image_id'].split(".png")[0]+"_change"+str(2*changes-1)+"".join(["_"+record for record_id, record in enumerate(caption['change_record']) if record_id<=(2*changes-1)])+".png") 
        if (idx+1)<=train_num:
            tgt_path = os.path.join(f"./{name}/TRAIN/imgs/2", str(idx)+".png")
        else:
            tgt_path = os.path.join(f"./{name}/TEST/imgs/2", str(idx-train_num)+".png")
        # tgt_path = os.path.join("./CLEVR/imgs/2", str(idx)+".png")
        shutil.copy2(source_path, tgt_path)
    
    # reverse captions (don't copy file--save space...)
    # past
    # for idx, caption in enumerate(captions_reverse):    
    #     # if idx == 2:
    #     #     break
    #     tmp_caption = []
    #     tmp_caplen = []    
    #     for change_idx, change_caption in enumerate(caption['change_captions']):
    #         end_flag = 0
    #         if change_idx==0:
    #             state="<start>"
    #         else:
    #             state="<sep>"
    #         if change_idx==len(caption['change_captions'])-1:
    #             end_flag=1
    #         t_caption, t_caplen = get_caption_json(change_caption,state, end_flag)          
    #         tmp_caption.append(t_caption)
    #         tmp_caplen.append(t_caplen)
    #     tmp_caplen, tmp_caplen = reverse_caption(tmp_caption, tmp_caplen)
    #     all_caption_reverse_json.extend(tmp_caption)
    #     all_caplen_reverse_json.extend(tmp_caplen)

    # now:
    for idx, caption in enumerate(captions_reverse):
        cap, caplen = get_caption_json_reverse(caption['change_captions'])
        all_caption_reverse_json.append(cap)
        all_caplen_reverse_json.append(caplen)

    # all_caplen_json.extend(all_caplen_reverse_json)
    # all_caption_json.extend(all_caption_reverse_json)
    # train_num = 6*train_num # captions per image = 6
    
    # train
    with open(f"{name}/TRAIN/caplens/caplens.json", 'w') as f:
        json.dump(all_caplen_json[:train_num], f)
    with open(f"{name}/TRAIN/captions/captions.json", 'w') as f:
        json.dump(all_caption_json[:train_num], f)
    with open(f"{name}/TRAIN/caplens/caplens_reverse.json", 'w') as f:
        json.dump(all_caplen_reverse_json[:train_num], f)
    with open(f"{name}/TRAIN/captions/captions_reverse.json", 'w') as f:
        json.dump(all_caption_reverse_json[:train_num], f)

    # test
    with open(f"{name}/TEST/caplens/caplens.json", 'w') as f:
        json.dump(all_caplen_json[train_num:], f)
    with open(f"{name}/TEST/captions/captions.json", 'w') as f:
        json.dump(all_caption_json[train_num:], f)
    with open(f"{name}/TEST/caplens/caplens_reverse.json", 'w') as f:
        json.dump(all_caplen_reverse_json[train_num:], f)
    with open(f"{name}/TEST/captions/captions_reverse.json", 'w') as f:
        json.dump(all_caption_reverse_json[train_num:], f)

    word_vac_json = {k:v for v,k in enumerate(word_vac)}
    with open(f"{name}/wordmap/wordmap.json", 'w') as f:
        json.dump(word_vac_json, f)

def generate_data_2_imgs(name1, changes):
    all_caption_json, all_caplen_json = [], []
    with open(change_caption_json_path, 'r')as f:
        captions = json.load(f)
    
    train_num = int(0.9*len(captions)*2)

    train_count, test_count = 0, 0
    
    for idx, caption in enumerate(captions):
        if changes==3:
            middle = int(len(caption['change_captions'])//2)
            end=len(caption['change_captions'])
        else:
            middle=1
            end=2
        cap, caplen = get_caption_json_2(caption['change_captions'][:middle])
        all_caption_json.append(cap)
        all_caplen_json.append(caplen)
        
        cap, caplen = get_caption_json_2(caption['change_captions'][middle:end])
        all_caption_json.append(cap)
        all_caplen_json.append(caplen)

        source_path_0_0 = os.path.join(os.path.join(data_dir, 'images'), caption['image_id'])
        source_path_0_1 = os.path.join(os.path.join(data_dir, 'images'), caption['image_id'].split(".png")[0]+"_change"+str(changes-1)+"".join(["_"+record for record_id, record in enumerate(caption['change_record']) if record_id<=(changes-1)])+".png")
        source_path_1_0 = source_path_0_1
        source_path_1_1 = os.path.join(os.path.join(data_dir, 'images'), caption['image_id'].split(".png")[0]+"_change"+str(2*changes-1)+"".join(["_"+record for record_id, record in enumerate(caption['change_record']) if record_id<=(2*changes-1)])+".png") 

        if (idx+1) <= train_num: # train data
            tgt_path_0_0 = os.path.join(f"{name1}/TRAIN/imgs/0", str(train_count)+".png")
            tgt_path_0_1 = os.path.join(f"{name1}/TRAIN/imgs/1", str(train_count)+".png")
            train_count+=1
            tgt_path_1_0 = os.path.join(f"{name1}/TRAIN/imgs/0", str(train_count)+".png")
            tgt_path_1_1 = os.path.join(f"{name1}/TRAIN/imgs/1", str(train_count)+".png")
            train_count+=1

        else: # test data
            tgt_path_0_0 = os.path.join(f"{name1}/TEST/imgs/0", str(test_count)+".png")
            tgt_path_1_0 = os.path.join(f"{name1}/TEST/imgs/1", str(test_count)+".png")
            test_count+=1
            tgt_path_0_1 = os.path.join(f"{name1}/TEST/imgs/0", str(test_count)+".png")
            tgt_path_1_1 = os.path.join(f"{name1}/TEST/imgs/1", str(test_count)+".png")
            test_count+=1
        
        shutil.copy2(source_path_0_0, tgt_path_0_0)
        shutil.copy2(source_path_0_1, tgt_path_0_1)
        shutil.copy2(source_path_1_0, tgt_path_1_0)
        shutil.copy2(source_path_1_1, tgt_path_1_1)
        
    with open(f"{name1}/TRAIN/caplens/caplens.json", 'w') as f:
        json.dump(all_caplen_json[:train_num], f)
    with open(f"{name1}/TRAIN/captions/captions.json", 'w') as f:
        json.dump(all_caption_json[:train_num], f)

    # test
    with open(f"{name1}/TEST/caplens/caplens.json", 'w') as f:
        json.dump(all_caplen_json[train_num:], f)
    with open(f"{name1}/TEST/captions/captions.json", 'w') as f:
        json.dump(all_caption_json[train_num:], f)

    word_vac_json = {k:v for v,k in enumerate(word_vac)}
    with open(f"{name1}/wordmap/wordmap.json", 'w') as f:
        json.dump(word_vac_json, f)

def generate_data_changes1(name,changes):
    assert changes==1
    all_caption_json, all_caplen_json, all_caption_reverse_json, all_caplen_reverse_json = [], [], [], []
    with open(change_caption_json_path, 'r')as f:
        captions = json.load(f)
    with open(change_caption_reverse_json_path, 'r') as f:
        captions_reverse = json.load(f)
    
    train_num = int(0.9*len(captions)*3) # 3 = 6/2, 6是因为之前是3幅图6个change，而/2是因为现在3幅图2个change
        
    for idx, caption in enumerate(captions): 
        # now, get captions like [[]pair 1, []pair 2...]
        cap, caplen = get_caption_json_2(caption['change_captions'])
        all_caption_json.append(cap)
        all_caplen_json.append(caplen)

        source_path = os.path.join(os.path.join(data_dir, 'images'), caption['image_id'])
        if (idx+1)<=train_num:
            tgt_path = os.path.join(f"./{name}/TRAIN/imgs/0", str(idx)+".png")
        else:
            tgt_path = os.path.join(f"./{name}/TEST/imgs/0", str(idx-train_num)+".png")
        # img 1
        shutil.copy2(source_path, tgt_path)
        # img 2
        source_path = os.path.join(os.path.join(data_dir, 'images'), caption['image_id'].split(".png")[0]+"_change"+str(changes-1)+"".join(["_"+record for record_id, record in enumerate(caption['change_record']) if record_id<=(changes-1)])+".png") 
        if (idx+1)<=train_num:
            tgt_path = os.path.join(f"./{name}/TRAIN/imgs/1", str(idx)+".png")
        else:
            tgt_path = os.path.join(f"./{name}/TEST/imgs/1", str(idx-train_num)+".png")
        shutil.copy2(source_path, tgt_path)
        # img 3
        source_path = os.path.join(os.path.join(data_dir, 'images'), caption['image_id'].split(".png")[0]+"_change"+str(2*changes-1)+"".join(["_"+record for record_id, record in enumerate(caption['change_record']) if record_id<=(2*changes-1)])+".png") 
        if (idx+1)<=train_num:
            tgt_path = os.path.join(f"./{name}/TRAIN/imgs/2", str(idx)+".png")
        else:
            tgt_path = os.path.join(f"./{name}/TEST/imgs/2", str(idx-train_num)+".png")
        # tgt_path = os.path.join("./CLEVR/imgs/2", str(idx)+".png")
        shutil.copy2(source_path, tgt_path)
    
    # reverse captions (don't copy file--save space...)
    # past
    # for idx, caption in enumerate(captions_reverse):    
    #     # if idx == 2:
    #     #     break
    #     tmp_caption = []
    #     tmp_caplen = []    
    #     for change_idx, change_caption in enumerate(caption['change_captions']):
    #         end_flag = 0
    #         if change_idx==0:
    #             state="<start>"
    #         else:
    #             state="<sep>"
    #         if change_idx==len(caption['change_captions'])-1:
    #             end_flag=1
    #         t_caption, t_caplen = get_caption_json(change_caption,state, end_flag)          
    #         tmp_caption.append(t_caption)
    #         tmp_caplen.append(t_caplen)
    #     tmp_caplen, tmp_caplen = reverse_caption(tmp_caption, tmp_caplen)
    #     all_caption_reverse_json.extend(tmp_caption)
    #     all_caplen_reverse_json.extend(tmp_caplen)

    # now:
    for idx, caption in enumerate(captions_reverse):
        cap, caplen = get_caption_json_reverse(caption['change_captions'])
        all_caption_reverse_json.append(cap)
        all_caplen_reverse_json.append(caplen)

    # all_caplen_json.extend(all_caplen_reverse_json)
    # all_caption_json.extend(all_caption_reverse_json)
    # train_num = 6*train_num # captions per image = 6
    
    # train
    with open(f"{name}/TRAIN/caplens/caplens.json", 'w') as f:
        json.dump(all_caplen_json[:train_num], f)
    with open(f"{name}/TRAIN/captions/captions.json", 'w') as f:
        json.dump(all_caption_json[:train_num], f)
    with open(f"{name}/TRAIN/caplens/caplens_reverse.json", 'w') as f:
        json.dump(all_caplen_reverse_json[:train_num], f)
    with open(f"{name}/TRAIN/captions/captions_reverse.json", 'w') as f:
        json.dump(all_caption_reverse_json[:train_num], f)

    # test
    with open(f"{name}/TEST/caplens/caplens.json", 'w') as f:
        json.dump(all_caplen_json[train_num:], f)
    with open(f"{name}/TEST/captions/captions.json", 'w') as f:
        json.dump(all_caption_json[train_num:], f)
    with open(f"{name}/TEST/caplens/caplens_reverse.json", 'w') as f:
        json.dump(all_caplen_reverse_json[train_num:], f)
    with open(f"{name}/TEST/captions/captions_reverse.json", 'w') as f:
        json.dump(all_caption_reverse_json[train_num:], f)

    word_vac_json = {k:v for v,k in enumerate(word_vac)}
    with open(f"{name}/wordmap/wordmap.json", 'w') as f:
        json.dump(word_vac_json, f)



# def extract_feature(name, split, id):
#     resnet50 = models.resnet50(pretrained=True)
#     resnet50 = nn.Sequential(*list(resnet50.children())[:-3])
#     resnet50.eval()
#     imgs = []
#     img_folder = os.path.join(f"{name}", split, "img", id)
#     transforms_to_tensor = transforms.Compose([
#     # transforms.Resize((16, 16)),
#     # transforms.ToTensor(),  # 转换图片为Tensor并归一化到[0, 1]
#     #     transforms.Resize(256),
#     # transforms.CenterCrop(224),
#     transforms.ToTensor(),
#     # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
# ])
#     for i in tqdm(range(len(os.listdir(img_folder))), 
#                   desc=f"extracting features of {img_folder}"):
#         img = Image.open(os.path.join(img_folder, str(i)+".png"))
#         img_tensor = transforms_to_tensor(img)
#         with torch.no_grad:
#             features = resnet50(img_tensor)
#         imgs.append(features.numpy())
#     h5_file = os.path.join(name, split, "imgs", str(id), "features.h5") 
#     with h5py.File(h5_file, 'w') as hf:
#         dataset = hf.ceate_dataset('features', shape=len(os.listdir(img_folder), )+imgs[0].shape[1:], dtype=imgs[0].dtype)
    
#         for idx, feature in enumerate(features):
#             dataset[idx]=feature
        

if __name__=="__main__":

    changes = 3 # changes between 2 imgs
    generate_data(name, changes=changes)
    generate_data_2_imgs(name1, changes=changes)
    
