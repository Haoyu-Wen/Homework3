import torch
from torch import nn
import math
from torch.nn.init import xavier_uniform_
from models_trans import CrossTransformer
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu
import torchvision.models as models

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# resnet101 = models.resnet101(pretrained=True)
# for param in resnet101.parameters():
#     param.requires_grad = False

resnet50 = models.resnet50(pretrained=True)
resnet50.eval()

# base requirement n=3
class MyMCCFormers_S(nn.Module):
  """
  MCCFormers-S
  """

  def __init__(self, feature_dim, h, w, d_model = 512, n_head = 4, n_layers = 2, dim_feedforward = 2048):
    """
    :param feature_dim: feature dimension of input dimension
    :param d_model: dimension of input to Transformer
    :param n_head: the number of heads in Transformer
    :param n_layers: the number of layers of Transformer
    :param dim_feedforward: dimension of hidden state
    :param h: height of input image
    :param w: width of input image
    """
    super(MyMCCFormers_S, self).__init__()
    self.resnet_50 = nn.Sequential(*list(resnet50.children())[:-3])

    self.input_proj = nn.Conv2d(feature_dim, d_model, kernel_size = 1)

    self.d_model = d_model

    encoder_layer = nn.TransformerEncoderLayer(d_model, n_head, dim_feedforward = dim_feedforward)
    self.transformer = nn.TransformerEncoder(encoder_layer, n_layers)
    # self.idx_embedding = nn.Embedding(2, d_model)
    self.idx_embedding = nn.Embedding(3, d_model) # modified by Navinvue
    self.w_embedding = nn.Embedding(w, int(d_model/2))
    self.h_embedding = nn.Embedding(h, int(d_model/2))
    self.h = h
    self.w = w

  def forward(self, img_feat1, img_feat2, img_feat3=None): # modified by Navinvue
    # img_feat1 (batch_size, feature_dim, h, w)
    batch = img_feat1.size(0)
    feature_dim = img_feat1.size(1)
    h, w = self.h, self.w

    d_model = self.d_model
    # print("$$$")
    # print(img_feat1.shape)
    img_feat1 = self.resnet_50(img_feat1)
    img_feat2 = self.resnet_50(img_feat2)
    img_feat3 = self.resnet_50(img_feat3)
    # print(img_feat1.shape)
    img_feat1 = self.input_proj(img_feat1)
    img_feat2 = self.input_proj(img_feat2)
    img_feat3 = self.input_proj(img_feat3) # modified by Navinvue
    # print(img_feat1.shape)
    img_feat1 = img_feat1.view(batch, d_model, -1) #(batch, d_model, h*w)
    img_feat2 = img_feat2.view(batch, d_model, -1) #(batch, d_model, h*w)
    img_feat3 = img_feat3.view(batch, d_model, -1) #(batch, d_model, h*w) modified by Navinvue

    # position embedding
    pos_w = torch.arange(w, device=device).to(device)
    pos_h = torch.arange(h, device=device).to(device)
    embed_w = self.w_embedding(pos_w)
    embed_h = self.h_embedding(pos_h)
    position_embedding = torch.cat([embed_w.unsqueeze(0).repeat(h, 1, 1),
                                    embed_h.unsqueeze(1).repeat(1, w, 1)], 
                                    dim = -1) 
    #(h, w, d_model)
    position_embedding = position_embedding.permute(2, 0, 1).unsqueeze(0).repeat(batch, 1, 1, 1) #(batch, d_model, h, w)
    position_embedding = position_embedding.view(batch, d_model, -1)

    # print(img_feat1.shape)
    # print(position_embedding.shape)
    img_feat1 = img_feat1 + position_embedding #(batch, d_model, h*w)
    img_feat2 = img_feat2 + position_embedding #(batch, d_model, h*w)
    img_feat3 = img_feat3 + position_embedding #(batch, d_model, h*w) #modified by Navinv

    # img_feat_cat = torch.cat([img_feat1, img_feat2], dim = 2) #(batch, d_model, 2*h*w)
    img_feat_cat = torch.cat([img_feat1, img_feat2, img_feat3], dim = 2) #(batch, d_model, 3*h*w) modified by Navinvue
    # img_feat_cat = img_feat_cat.permute(2, 0, 1) #(2*h*w, batch, d_model)
    img_feat_cat = img_feat_cat.permute(2, 0, 1) #(3*h*w, batch, d_model) #modified by Navinvue
    
    # idx = 0, 1 for img_feat1, img_feat2, respectively
    idx1 = torch.zeros(batch, h*w).long().to(device)
    idx2 = torch.ones(batch, h*w).long().to(device)
    idx3 = torch.full((batch, h*w), 2).long().to(device) # modified by Navinvue #Undo, torch.zeros?ones?2s?
    # idx = torch.cat([idx1, idx2], dim = 1) #(batch, 2*h*w)
    idx = torch.cat([idx1, idx2, idx3], dim = 1) #(batch, 3*h*w) modified by Navinvue
    # idx_embedding = self.idx_embedding(idx) #(batch, 2*h*w, d_model)
    idx_embedding = self.idx_embedding(idx) #(batch, 3*h*w, d_model) modified by Navinvue
    idx_embedding = idx_embedding.permute(1, 0, 2) #(2*h*w, batch, d_model) #(3*h*w, batch, d_model) modified by Navinvue

    feature = img_feat_cat + idx_embedding #(2*h*w, batch, d_model) #(3*h*w, batch, d_model) modified by Navinvue
    feature = self.transformer(feature) #(2*h*w, batch, d_model) #(3*h*w, batch, d_model) modified by Navinvue

    img_feat1 = feature[:h*w].permute(1, 2, 0)  #(batch, d_model, h*w)
    img_feat1 = img_feat1.view(batch, d_model, -1).permute(2, 0, 1) #(batch, d_model, h*w)
    # img_feat2 = feature[h*w:].permute(1, 2, 0)  #(batch, d_model, h*w)
    img_feat2 = feature[h*w:2*h*w].permute(1, 2, 0)  #(batch, d_model, h*w) #modified by Navinvue
    img_feat2 = img_feat2.view(batch, d_model, -1).permute(2, 0, 1) #(batch, d_model, h*w)
    # start modify by Navinvue
    img_feat3 = feature[2*h*w:].permute(1, 2, 0)  #(batch, d_model, h*w) #modified by Navinvue
    img_feat3 = img_feat3.view(batch, d_model, -1).permute(2, 0, 1) #(batch, d_model, h*w)
    # end modify by Navinvue

    # img_feat = torch.cat([img_feat1,img_feat2],dim=2)
    img_feat = torch.cat([img_feat1, img_feat2, img_feat3], dim=2) # modified by Navinvue

    return img_feat

#advance(maybe) MCCFormers_S, input=2/3, use 2 imgs to train
# in fact, just encoder twice, no change in network 
class AdvanceMCCFormers_S(nn.Module):
  """
  AdvanceMCCFormers-S
  train for 2, test for n
  """

  def __init__(self, feature_dim, h, w, d_model = 512, n_head = 4, n_layers = 2, dim_feedforward = 2048):
    """
    :param feature_dim: feature dimension of input dimension
    :param d_model: dimension of input to Transformer
    :param n_head: the number of heads in Transformer
    :param n_layers: the number of layers of Transformer
    :param dim_feedforward: dimension of hidden state
    :param h: height of input image
    :param w: width of input image
    """
    super(AdvanceMCCFormers_S, self).__init__()

    self.input_proj = nn.Conv2d(feature_dim, d_model, kernel_size = 1)
    self.resnet_50 = nn.Sequential(*list(resnet50.children())[:-3])
    self.d_model = d_model
    self.h = h
    self.w = w 
    encoder_layer = nn.TransformerEncoderLayer(d_model, n_head, dim_feedforward = dim_feedforward)
    self.transformer = nn.TransformerEncoder(encoder_layer, n_layers)
    self.idx_embedding = nn.Embedding(3, d_model)
    self.w_embedding = nn.Embedding(w, int(d_model/2))
    self.h_embedding = nn.Embedding(h, int(d_model/2))

  def forward(self, img_feat1, img_feat2, img_feat3=None):
    # img_feat1 (batch_size, feature_dim, h, w)
    batch = img_feat1.size(0)
    feature_dim = img_feat1.size(1)
    # h, w = img_feat1.size(2), img_feat1.size(3)
    h = self.h
    w = self.w

    d_model = self.d_model
    img_feat1 = self.resnet_50(img_feat1)
    img_feat2 = self.resnet_50(img_feat2)
    if img_feat3 is not None:
      img_feat3 = self.resnet_50(img_feat3)

    img_feat1 = self.input_proj(img_feat1)
    img_feat2 = self.input_proj(img_feat2)
    img_feat1 = img_feat1.view(batch, d_model, -1) #(batch, d_model, h*w)
    img_feat2 = img_feat2.view(batch, d_model, -1) #(batch, d_model, h*w)
    if img_feat3 is not None:
      img_feat3 = self.input_proj(img_feat3)
      img_feat3 = img_feat3.view(batch, d_model, -1) #(batch, d_model, h*w)

    # position embedding
    pos_w = torch.arange(w, device=device).to(device)
    pos_h = torch.arange(h, device=device).to(device)
    embed_w = self.w_embedding(pos_w)
    embed_h = self.h_embedding(pos_h)
    position_embedding = torch.cat([embed_w.unsqueeze(0).repeat(h, 1, 1),
                                    embed_h.unsqueeze(1).repeat(1, w, 1)], 
                                    dim = -1) 
    #(h, w, d_model)
    position_embedding = position_embedding.permute(2, 0, 1).unsqueeze(0).repeat(batch, 1, 1, 1) #(batch, d_model, h, w)
    position_embedding = position_embedding.view(batch, d_model, -1)

    img_feat1 = img_feat1 + position_embedding #(batch, d_model, h*w)
    img_feat2 = img_feat2 + position_embedding #(batch, d_model, h*w)
    if img_feat3 is not None:
      img_feat3 = img_feat3 + position_embedding #(batch, d_model, h*w)

    img_feat_cat = torch.cat([img_feat1, img_feat2], dim = 2) #(batch, d_model, 2*h*w)
    img_feat_cat = img_feat_cat.permute(2, 0, 1) #(2*h*w, batch, d_model)
    if img_feat3 is not None:
      img_feat_cat_hat = torch.cat([img_feat2, img_feat3], dim = 2)
      img_feat_cat_hat = img_feat_cat_hat.permute(2, 0, 1) 

    # idx = 0, 1 for img_feat1, img_feat2, respectively
    idx1 = torch.zeros(batch, h*w).long().to(device)
    idx2 = torch.ones(batch, h*w).long().to(device)
    idx = torch.cat([idx1, idx2], dim = 1) #(batch, 2*h*w)
    idx_embedding = self.idx_embedding(idx) #(batch, 2*h*w, d_model)
    idx_embedding = idx_embedding.permute(1, 0, 2) #(2*h*w, batch, d_model)
    if img_feat3 is not None:
      idx3 = torch.full((batch, h*w), 2).long().to(device)
      idx_hat = torch.cat([idx2, idx3], dim=1)
      idx_embedding_hat = self.idx_embedding(idx_hat)
      idx_embedding_hat = idx_embedding_hat.permute(1, 0, 2)

    feature = img_feat_cat + idx_embedding #(2*h*w, batch, d_model)
    feature = self.transformer(feature) #(2*h*w, batch, d_model)

    if img_feat3 is not None:
      feature_hat = img_feat_cat_hat + idx_embedding_hat
      feature_hat = self.transformer(feature_hat)

    if img_feat3 is not None:
      img_feat3 =feature_hat[h*w:].permute(1, 2, 0)
      img_feat3 = img_feat3.view(batch, d_model, -1).permute(2, 0, 1)
      img_feat2 = feature_hat[:h*w].permute(1, 2, 0)
      img_feat2 = 0.5*(img_feat2+feature[h*w:].permute(1, 2, 0))
      img_feat2 = img_feat2.view(batch, d_model, -1).permute(2, 0, 1)
    if img_feat3 is None:
      img_feat2 = feature[h*w:].permute(1, 2, 0)  #(batch, d_model, h*w)
      img_feat2 = img_feat2.view(batch, d_model, -1).permute(2, 0, 1) #(batch, d_model, h*w)

    img_feat1 = feature[:h*w].permute(1, 2, 0)  #(batch, d_model, h*w)
    img_feat1 = img_feat1.view(batch, d_model, -1).permute(2, 0, 1) #(batch, d_model, h*w)

    img_feat = torch.cat([img_feat1,img_feat2],dim=2)
    if img_feat3 is not None:
      img_feat_hat = torch.cat([img_feat2, img_feat3], dim=2)
      # return 0.5*img_feat + 0.5*img_feat_hat
      return img_feat, img_feat_hat
    return img_feat

# base requirement n=3
#method 1, MyMCCFormers_D_1 (1)Q img1, KV(img2) (2) Q img 2, KV(img1) (3) Q img 3, KV(img2) (4) Q img 2, KV(img3)
#method 2, MyMCCFormers_D_2 (1)Q img1, KV(img2, img3) (2) Q img2, KV(img1, img3) (3) Q img3, KV(img1, img2)
class MyMCCFormers_D_1(nn.Module):
  """
  MyMCCFormers-D-1
  (1)Q img1, KV(img2) (2) Q img 2, KV(img1) (3) Q img 3, KV(img2) (4) Q img 2, KV(img3)
  """
  def __init__(self, feature_dim, dropout, h, w, d_model = 512, n_head = 4, n_layers = 2):
    """
    :param feature_dim: dimension of input features
    :param dropout: dropout rate
    :param d_model: dimension of hidden state
    :param n_head: number of heads in multi head attention
    :param n_layer: number of layers of transformer layer
    """
    super(MyMCCFormers_D_1, self).__init__()
    self.d_model = d_model
    self.n_layers = n_layers
    self.resnet_50 = nn.Sequential(*list(resnet50.children())[:-3])
    self.w_embedding = nn.Embedding(w, int(d_model/2))
    self.h_embedding = nn.Embedding(h, int(d_model/2))

    self.projection = nn.Conv2d(feature_dim, d_model, kernel_size = 1)
    self.transformer = nn.ModuleList([CrossTransformer(dropout, d_model, n_head) for i in range(n_layers)])
    self.h = h
    self.w = w
    self._reset_parameters()

  def _reset_parameters(self):
    """Initiate parameters in the transformer model."""        
    for p in self.parameters():
      if p.dim() > 1:
        xavier_uniform_(p)

  def forward(self, img_feat1, img_feat2, img_feat3=None):
    # img_feat1 (batch_size, feature_dim, h, w)
    batch = img_feat1.size(0)
    feature_dim = img_feat1.size(1)
    w, h = self.w, self.h
    img_feat1 = self.resnet_50(img_feat1)
    img_feat2 = self.resnet_50(img_feat2)
    img_feat3 = self.resnet_50(img_feat3)

    img_feat1 = self.projection(img_feat1)# + position_embedding # (batch_size, d_model, h, w)
    img_feat2 = self.projection(img_feat2)# + position_embedding # (batch_size, d_model, h, w)
    img_feat3 = self.projection(img_feat3)# + position_embedding # (batch_size, d_model, h, w)

    pos_w = torch.arange(w,device=device).to(device)
    pos_h = torch.arange(h,device=device).to(device)
    embed_w = self.w_embedding(pos_w)
    embed_h = self.h_embedding(pos_h)
    position_embedding = torch.cat([embed_w.unsqueeze(0).repeat(h, 1, 1),
                                   embed_h.unsqueeze(1).repeat(1, w, 1)], 
                                   dim = -1) 
    #(h, w, d_model)
    position_embedding = position_embedding.permute(2, 0, 1).unsqueeze(0).repeat(batch, 1, 1, 1) #(batch, d_model, h, w)

    img_feat1 = img_feat1 + position_embedding # (batch_size, d_model, h, w)
    img_feat2 = img_feat2 + position_embedding # (batch_size, d_model, h, w)
    img_feat3 = img_feat3 + position_embedding # (batch_size, d_model, h, w)

    output1 = img_feat1.view(batch, self.d_model, -1).permute(2, 0, 1) # (h*w, batch_size, d_model)
    output2 = img_feat2.view(batch, self.d_model, -1).permute(2, 0, 1) # (h*w, batch_size, d_model)
    output3 = img_feat3.view(batch, self.d_model, -1).permute(2, 0, 1) # (h*w, batch_size, d_model)

    for l in self.transformer:
      output1_2, output2_1, output2_3, output3_2 = l(output1, output2), l(output2, output1), l(output2, output3), l(output3, output2)


    position_embedding = position_embedding.view(batch,self.d_model,-1).permute(2,0,1)
    output1 = output1_2 #+ position_embedding
    output2 = output2_1 #+ position_embedding
    output3 = output2_3
    output4 = output3_2


    output = torch.cat([output1,output2,output3,output4],dim=2)
    #output1 = output1.permute(1, 2, 0).view(batch,512,16,16) #(batch_size, d_model, h*w)
    #output2 = output2.permute(1, 2, 0).view(batch,512,16,16) #(batch_size, d_model, h*w)
    return output

#method 2, MyMCCFormers_D_2 (1)Q img1, KV(img2, img3) (2) Q img2, KV(img1, img3) (3) Q img3, KV(img1, img2)
class MyMCCFormers_D_2(nn.Module):
  """
  MyMCCFormers-D-2
  """
  def __init__(self, feature_dim, dropout, h, w, d_model = 512, n_head = 4, n_layers = 2):
    """
    :param feature_dim: dimension of input features
    :param dropout: dropout rate
    :param d_model: dimension of hidden state
    :param n_head: number of heads in multi head attention
    :param n_layer: number of layers of transformer layer
    """
    super(MyMCCFormers_D_2, self).__init__()
    self.d_model = d_model
    self.n_layers = n_layers
    self.resnet_50 = nn.Sequential(*list(resnet50.children())[:-3])
    self.w_embedding = nn.Embedding(w, int(d_model/2))
    self.h_embedding = nn.Embedding(h, int(d_model/2))

    self.projection = nn.Conv2d(feature_dim, d_model, kernel_size = 1)
    self.transformer = nn.ModuleList([CrossTransformer(dropout, d_model, n_head) for i in range(n_layers)])
    self.h = h
    self.w = w
    self._reset_parameters()

  def _reset_parameters(self):
    """Initiate parameters in the transformer model."""        
    for p in self.parameters():
      if p.dim() > 1:
        xavier_uniform_(p)

  def forward(self, img_feat1, img_feat2, img_feat3=None):
    # img_feat1 (batch_size, feature_dim, h, w)
    batch = img_feat1.size(0)
    feature_dim = img_feat1.size(1)
    # w, h = img_feat1.size(2), img_feat1.size(3)
    w = self.w 
    h = self.h

    img_feat1 = self.resnet_50(img_feat1)
    img_feat2 = self.resnet_50(img_feat2)
    img_feat3 = self.resnet_50(img_feat3)

    img_feat1 = self.projection(img_feat1)# + position_embedding # (batch_size, d_model, h, w)
    img_feat2 = self.projection(img_feat2)# + position_embedding # (batch_size, d_model, h, w)
    img_feat3 = self.projection(img_feat3)# + position_embedding # (batch_size, d_model, h, w)


    pos_w = torch.arange(w,device=device).to(device)
    pos_h = torch.arange(h,device=device).to(device)
    embed_w = self.w_embedding(pos_w)
    embed_h = self.h_embedding(pos_h)
    position_embedding = torch.cat([embed_w.unsqueeze(0).repeat(h, 1, 1),
                                   embed_h.unsqueeze(1).repeat(1, w, 1)], 
                                   dim = -1) 
    #(h, w, d_model)
    position_embedding = position_embedding.permute(2, 0, 1).unsqueeze(0).repeat(batch, 1, 1, 1) #(batch, d_model, h, w)

    img_feat1 = img_feat1 + position_embedding # (batch_size, d_model, h, w)
    img_feat2 = img_feat2 + position_embedding # (batch_size, d_model, h, w)
    img_feat3 = img_feat3 + position_embedding # (batch_size, d_model, h, w)

    output1 = img_feat1.view(batch, self.d_model, -1).permute(2, 0, 1) # (h*w, batch_size, d_model)
    output2 = img_feat2.view(batch, self.d_model, -1).permute(2, 0, 1) # (h*w, batch_size, d_model)
    output3 = img_feat3.view(batch, self.d_model, -1).permute(2, 0, 1) # (h*w, batch_size, d_model)
    
    output_1_2 = torch.cat([output1, output2], dim=0) # (2*h*w, batch_size, d_model)
    output_1_3 = torch.cat([output1, output3], dim=0) # (2*h*w, batch_size, d_model)
    output_2_3 = torch.cat([output2, output3], dim=0) # (2*h*w, batch_size, d_model)

    for l in self.transformer:
      output1, output2, output3 = l(output1, output_2_3), l(output2, output_1_3), l(output3, output_1_2)


    position_embedding = position_embedding.view(batch,self.d_model,-1).permute(2,0,1)
    output1 = output1 #+ position_embedding
    output2 = output2 #+ position_embedding
    output3 = output3 #+ position_embedding

    output = torch.cat([output1, output2, output3],dim=2)
    #output1 = output1.permute(1, 2, 0).view(batch,512,16,16) #(batch_size, d_model, h*w)
    #output2 = output2.permute(1, 2, 0).view(batch,512,16,16) #(batch_size, d_model, h*w)
    return output

# advance?
class AdvanceMCCFormers_D(nn.Module):
  """
  AdvanceMCCFormers-D
  """
  def __init__(self, feature_dim, dropout, h, w, d_model = 512, n_head = 4, n_layers = 2):
    """
    :param feature_dim: dimension of input features
    :param dropout: dropout rate
    :param d_model: dimension of hidden state
    :param n_head: number of heads in multi head attention
    :param n_layer: number of layers of transformer layer
    """
    super(AdvanceMCCFormers_D, self).__init__()
    self.d_model = d_model
    self.n_layers = n_layers
    self.resnet_50 = nn.Sequential(*list(resnet50.children())[:-3])
    self.w_embedding = nn.Embedding(w, int(d_model/2))
    self.h_embedding = nn.Embedding(h, int(d_model/2))

    self.projection = nn.Conv2d(feature_dim, d_model, kernel_size = 1)
    self.transformer = nn.ModuleList([CrossTransformer(dropout, d_model, n_head) for i in range(n_layers)])
    
    self.h=h
    self.w=w
    self._reset_parameters()

  def _reset_parameters(self):
    """Initiate parameters in the transformer model."""        
    for p in self.parameters():
      if p.dim() > 1:
        xavier_uniform_(p)

  def forward(self, img_feat1, img_feat2, img_feat3=None):
    # img_feat1 (batch_size, feature_dim, h, w)
    batch = img_feat1.size(0)
    feature_dim = img_feat1.size(1)
    # w, h = img_feat1.size(2), img_feat1.size(3)
    w = self.w
    h = self.h

    img_feat1 = self.resnet_50(img_feat1)
    img_feat2 = self.resnet_50(img_feat2)
    if img_feat3 is not None:
      img_feat3 = self.resnet_50(img_feat3)

    img_feat1 = self.projection(img_feat1)# + position_embedding # (batch_size, d_model, h, w)
    img_feat2 = self.projection(img_feat2)# + position_embedding # (batch_size, d_model, h, w)
    if img_feat3 is not None:
      img_feat3 = self.projection(img_feat3)

    pos_w = torch.arange(w,device=device).to(device)
    pos_h = torch.arange(h,device=device).to(device)
    embed_w = self.w_embedding(pos_w)
    embed_h = self.h_embedding(pos_h)
    position_embedding = torch.cat([embed_w.unsqueeze(0).repeat(h, 1, 1),
                                   embed_h.unsqueeze(1).repeat(1, w, 1)], 
                                   dim = -1) 
    #(h, w, d_model)
    position_embedding = position_embedding.permute(2, 0, 1).unsqueeze(0).repeat(batch, 1, 1, 1) #(batch, d_model, h, w)

    img_feat1 = img_feat1 + position_embedding # (batch_size, d_model, h, w)
    img_feat2 = img_feat2 + position_embedding # (batch_size, d_model, h, w)
    if img_feat3 is not None:
      img_feat3 = img_feat3 + position_embedding

    output1 = img_feat1.view(batch, self.d_model, -1).permute(2, 0, 1) # (h*w, batch_size, d_model)
    output2 = img_feat2.view(batch, self.d_model, -1).permute(2, 0, 1) # (h*w, batch_size, d_model)

    if img_feat3 is not None:
      output2_hat = img_feat2.view(batch, self.d_model, -1).permute(2, 0, 1) # (h*w, batch_size, d_model)
      output3 = img_feat3.view(batch, self.d_model, -1).permute(2, 0, 1) # (h*w, batch_size, d_model)

    for l in self.transformer:
      output1, output2 = l(output1, output2), l(output2, output1)
      if img_feat3 is not None:
        output2_hat, output3 = l(output2_hat, output3), l(output3, output2_hat)
        


    position_embedding = position_embedding.view(batch,self.d_model,-1).permute(2,0,1)
    output1 = output1 #+ position_embedding
    output2 = output2 #+ position_embedding
    if img_feat3 is not None:
      output2_hat = output2_hat
      output3 = output3


    output = torch.cat([output1,output2],dim=2)
    if img_feat3 is not None:
      output_hat = torch.cat([output2_hat, output3], dim=2)
      return output, output_hat
    #output1 = output1.permute(1, 2, 0).view(batch,512,16,16) #(batch_size, d_model, h*w)
    #output2 = output2.permute(1, 2, 0).view(batch,512,16,16) #(batch_size, d_model, h*w)
    return output
