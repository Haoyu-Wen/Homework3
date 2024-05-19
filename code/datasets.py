import torch
from torch.utils.data import Dataset
# import h5py
import json
import os
from PIL import Image # modified by Navinvue, read png pictures
import torchvision.transforms as transforms # modified by Navinvue
import torch.nn as nn
# padding_len = 120


class CaptionDataset(Dataset):
  def __init__(self, data_folder, data_name, split, captions_per_image, dataset_name):
    self.split = split
    assert self.split in {'TRAIN', 'VAL', 'TEST'}
    self.h1 = h5py.File(os.path.join(data_folder, self.split + '_IMAGE_FEATURES_1_' + data_name + '.h5'), 'r')
    self.imgs1 = self.h1['images_features']

    self.h2 = h5py.File(os.path.join(data_folder, self.split + '_IMAGE_FEATURES_2_' + data_name + '.h5'), 'r')
    self.imgs2 = self.h2['images_features']

    self.cpi = captions_per_image

    self.dataset_name = dataset_name

    with open(os.path.join(data_folder, self.split + '_CAPTIONS_' + data_name + '.json'), 'r') as f:
      self.captions = json.load(f)

    with open(os.path.join(data_folder, self.split + '_CAPLENS_' + data_name + '.json'), 'r') as f:
      self.caplens = json.load(f)

    self.dataset_size = len(self.captions)

  def __getitem__(self, i):

    if self.dataset_name == 'MOSCC':
      img1 = torch.FloatTensor(self.imgs1[i // self.cpi])
      img2 = torch.FloatTensor(self.imgs2[i // self.cpi])

    if self.dataset_name == 'CCHANGE' or self.dataset_name == 'STD':
      img1 = torch.FloatTensor(self.imgs1[i])
      img2 = torch.FloatTensor(self.imgs2[i])

    caption = torch.LongTensor(self.captions[i])
    caplen = torch.LongTensor([self.caplens[i]])

    if self.split is 'TRAIN':
      return img1, img2, caption, caplen
    else:
      if self.dataset_name == 'MOSCC':
        all_captions = torch.LongTensor(
          self.captions[((i // self.cpi) * self.cpi):((i//self.cpi)*self.cpi) + self.cpi])
        return img1, img2, caption, caplen, all_captions
      if self.dataset_name == 'CCHANGE' or self.dataset_name == 'STD':
        return img1, img2, caption, caplen, caption

  def __len__(self):
    return self.dataset_size

class ModifiedCaptionDataset(Dataset): #modified by Navinvue
  def __init__(self, data_folder, data_name, split, captions_per_image, dataset_name, wordmap=None, num_images=3, encode_twice=False, reverse_data=False):
    self.split = split
    assert self.split in {'TRAIN', 'VAL', 'TEST'}
    
    self.cpi = captions_per_image
    self.dataset_name = dataset_name # dataset name，find data in data_folder/dataset_name/TRAIN(TEST)/imgs(captions or caplens)
    self.num_images = num_images # 两相图片还是多项（3项）
    self.encode_twice = encode_twice

    img_folder = os.path.join(data_folder, self.dataset_name, self.split ,'imgs')
    img1_folder = os.path.join(img_folder,'0') # before
    img2_folder = os.path.join(img_folder, '1') # during change
    img3_folder = os.path.join(img_folder, '2') # after
    captions_folder = os.path.join(data_folder, self.dataset_name, self.split, 'captions')
    caplen_folder = os.path.join(data_folder, self.dataset_name, self.split, 'caplens')
    imgs1, imgs2, imgs3 = [], [], []
    
    # transforms jpg/png to torch tensor
    transform_to_tensor = transforms.Compose([
    # transforms.Resize((16, 16)),
    # transforms.ToTensor(),  # 转换图片为Tensor并归一化到[0, 1]
    #     transforms.Resize(256),
    # transforms.CenterCrop(224),
    transforms.ToTensor(),
    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
    
    for i in range(len(os.listdir(img1_folder))):
      imgs1.append(transform_to_tensor(Image.open(os.path.join(img1_folder, str(i) + '.png')).convert('RGB')))
      # imgs1.append(transform_to_tensor(Image.open(os.path.join(img1_folder, str(i) + '.png'))))
      # imgs1.append(torch.FloatTensor(Image.open(os.path.join(img1_folder, str(i) + '.png'))))
    for i in range(len(os.listdir(img2_folder))):
      # imgs2.append(torch.FloatTensor(Image.open(os.path.join(img1_folder, str(i) + '.png'))))
      imgs2.append(transform_to_tensor(Image.open(os.path.join(img2_folder, str(i) + '.png')).convert('RGB')))
      # imgs2.append(transform_to_tensor(Image.open(os.path.join(img2_folder, str(i) + '.png'))))
    if self.num_images == 3:
      for i in range(len(os.listdir(img3_folder))):
        # imgs3.append(torch.FloatTensor(Image.open(os.path.join(img1_folder, str(i) + '.png'))))
        imgs3.append(transform_to_tensor(Image.open(os.path.join(img3_folder, str(i) + '.png')).convert('RGB')))
        # imgs3.append(transform_to_tensor(Image.open(os.path.join(img3_folder, str(i) + '.png'))))
    else:
      imgs3 = [None for i in range(len(imgs1))]

    self.imgs1 = imgs1
    self.imgs2 = imgs2
    self.imgs3 = imgs3
    if reverse_data:
      # assert num_images==3
      if num_images == 3:
        self.imgs1.extend(imgs3)
        self.imgs2.extend(imgs2)
        self.imgs3.extend(imgs1)    
      else:
        self.imgs1.extend(imgs2)
        self.imgs2.extend(imgs1)
        self.imgs3.extend(imgs3)    
    
    captions, caplens = [], []
    
    with open(os.path.join(captions_folder, 'captions.json'), 'r') as f:
      captions=json.load(f)

    with open(os.path.join(caplen_folder, 'caplens.json'), 'r') as f:
      caplens = json.load(f)

    if reverse_data:
      with open(os.path.join(captions_folder, 'captions_reverse.json'), 'r') as f:
        captions.extend(json.load(f)) 

      with open(os.path.join(caplen_folder, 'caplens_reverse.json'), 'r') as f:
        caplens.extend(json.load(f))

    #弃用，对于两相图像训练，重新预处理为{dataset_name}+"hat"了，caption和caplen也对应修改了的
    if dataset_name=="CLEVR" and num_images==2: # 需要抛弃掉一些caption(因为原数据是3幅图的变化，6个caption，两幅图则是3个caption)
        t_captions, t_caplens = [], []
        for i in range(0, len(captions), 2*self.cpi):
          tt_captions = captions[i:i+self.cpi]
          tt_captions[-1].pop()
          tt_captions[-1].insert(tt_captions[-1].index(wordmap['<pad>']), wordmap['<end>'])
          tt_caplens = caplens[i:i+self.cpi]
          tt_caplens[-1]=tt_caplens[-1]-1
          t_captions.extend(tt_captions)
          t_caplens.extend(tt_caplens)
        captions, caplens = t_captions, t_caplens

    if encode_twice:
      assert num_images==3
      assert self.cpi==1
      t_captions, t_caplens = [[], []], [[], []] #假定3幅图，也即encode twice（相邻两幅之间）
      for i in range(0, len(captions)):
        t_caption = captions[i]
        count = 0
        for _, _word in enumerate(t_caption):
          if _word == wordmap['<sep>']:
            count+=1
          if count == 2: # 6 change
            break
          
        the_second_seq_index = _
        t_caption1, t_caption2= t_caption[:the_second_seq_index], t_caption[the_second_seq_index:]
        t_caption2 = t_caption2[:t_caption2.index(wordmap['<end>'])+1]
        t_caption1.append(wordmap['<end>'])
        t_caplen1 = len(t_caption1)
        t_caption2.pop() # remove the beginning sep
        t_caption2.insert(0, wordmap['<start>'])
        t_caplen2 = len(t_caption2)
        while len(t_caption1) < int(len(captions[i])):
          t_caption1.append(wordmap['<pad>'])
        while len(t_caption2) < int(len(captions[i])):
          t_caption2.append(wordmap['<pad>'])
        t_captions[0].append(t_caption1)
        t_captions[1].append(t_caption2)
        t_caplens[0].append(t_caplen1)
        t_caplens[1].append(t_caplen2)

      # for i in range(0, len(captions), 2 * self.cpi):

        # for j in range(self.cpi):
        #   tt = captions[i+j]
        #   tt[-1].remove(wordmap['<pad>'])
        #   tt[-1].insert(tt[-1].index('<pad>'), wordmap['<end>'])
        #   t_captions[0].append(tt)
        #   tt = captions[i+j+self.cpi]
        #   tt[0].remove(wordmap['<sep>'])
        #   tt[0].append(wordmap['<start>'])
        #   t_captions[1].append(tt)
        #   tt = caplens[i+j]+1
        #   t_caplens[0].append(tt)
        #   tt = caplens[i+j+self.cpi]
        #   t_caplens[1].append(tt)
      self.captions = t_captions
      self.caplens = t_caplens

    else:
      self.captions = captions
      self.caplens = caplens

    self.dataset_size = len(self.captions) if not self.encode_twice else len(self.captions[0])
    

  def __getitem__(self, i):

    # if self.dataset_name == 'MOSCC':
    #   img1 = torch.FloatTensor(self.imgs1[i // self.cpi])
    #   img2 = torch.FloatTensor(self.imgs2[i // self.cpi])

    # if self.dataset_name == 'CCHANGE' or self.dataset_name == 'STD':
    #   img1 = torch.FloatTensor(self.imgs1[i])
    #   img2 = torch.FloatTensor(self.imgs2[i])
    

    img1 = self.imgs1[i//self.cpi]
    img2 = self.imgs2[i//self.cpi]
    img3 = self.imgs3[i//self.cpi]
    
    # if self.encode_twice:
    #   caption1, caption2 = torch.LongTensor(self.captions[i]), torch.LongTensor([self.captions[i+self.cpi]])
    #   caplen1, caplen2 =  torch.LongTensor([self.caplens[i]]), torch.LongTensor([self.caplens[i+self.cpi]])
    # else:
      #   caption = torch.LongTensor(self.captions[i])
      #   caplen = torch.LongTensor([self.caplens[i]])
    if self.split == 'TRAIN':
      if self.encode_twice:
        if img3 is not None:
          return img1, img2, img3, torch.LongTensor(self.captions[0][i]), torch.LongTensor(self.captions[1][i]), torch.LongTensor([self.caplens[0][i]]), torch.LongTensor([self.caplens[1][i]]) 
        else:
          return img1, img2, torch.LongTensor(self.captions[0][i]), torch.LongTensor(self.captions[1][i]), torch.LongTensor([self.caplens[0][i]]), torch.LongTensor([self.caplens[1][i]]) 
      else:
        if img3 is not None:
          return img1, img2, img3, torch.LongTensor(self.captions[i]), torch.LongTensor([self.caplens[i]])
        else:
          return img1, img2, torch.LongTensor(self.captions[i]), torch.LongTensor([self.caplens[i]])
    else: # TEST, VALID
      if self.encode_twice:
        all_caps1 = torch.LongTensor(
          self.captions[0][((i // self.cpi) * self.cpi):((i//self.cpi)*self.cpi) + self.cpi])
        all_caps2 = torch.LongTensor(
          self.captions[1][((i // self.cpi) * self.cpi):((i//self.cpi)*self.cpi) + self.cpi])
        if img3 is not None:
          return img1, img2, img3, torch.LongTensor(self.captions[0][i]), torch.LongTensor(self.captions[1][i]), torch.LongTensor([self.caplens[0][i]]), torch.LongTensor([self.caplens[1][i]]), all_caps1, all_caps2
        else: # in fact impossiable
          return img1, img2, torch.LongTensor(self.captions[0][i]), torch.LongTensor(self.captions[1][i]), torch.LongTensor([self.caplens[0][i]]), torch.LongTensor([self.caplens[1][i]]), all_caps1, all_caps2
      else:
        all_caps = torch.LongTensor(
          self.captions[((i // self.cpi) * self.cpi):((i//self.cpi)*self.cpi) + self.cpi])
        if img3 is not None:
          return img1, img2, img3, torch.LongTensor(self.captions[i]), torch.LongTensor([self.caplens[i]]), all_caps 
        else:
          return img1, img2, torch.LongTensor(self.captions[i]), torch.LongTensor([self.caplens[i]]), all_caps 

    # caplen = torch.LongTensor([self.caplens[i]])

  def __len__(self):
    return self.dataset_size


