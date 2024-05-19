import json
import time
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.models as models

from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
from models_trans import MCCFormers_D, MCCFormers_S, DecoderTransformer, PlainDecoder
from modified_models_trans import MyMCCFormers_S, AdvanceMCCFormers_S, MyMCCFormers_D_1, MyMCCFormers_D_2, AdvanceMCCFormers_D
from datasets import *
from utils import *
from nltk.translate.bleu_score import corpus_bleu ##--
 

from torch.autograd import Variable

import argparse

# Data parameters
data_name = '3dcc_5_cap_per_img_0_min_word_freq'
# data_name = 
# Model parameters 
embed_dim = 512
decoder_dim = 512
dropout = 0.5
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
cudnn.benchmark = True
captions_per_image = 6


# Training parameters
start_epoch = 0
batch_size = 32
workers = 1
decoder_lr = 1e-4
encoder_lr = 1e-4
grap_clip = 5.
best_bleu4 = 0.
print_freq = 100

para_lambda1 = 1.0
para_lambda2 = 1.0

def get_key(dict_, value):
  return [k for k, v in dict_.items() if v == value]

def main(args):

  torch.manual_seed(0)
  torch.cuda.manual_seed_all(0)
  torch.backends.cudnn.benckmark = False
  torch.backends.cudnn.deterministic = True

  global start_epoch, data_name

  # Read word map
  # word_map_file = os.path.join(args.data_folder, 'WORDMAP_' + data_name + '.json')
  # modified by navinvue
  # word_map_file = os.path.join(args.data_folder, 'WORDMAP_' + data_name + '.json')
  word_map_file = os.path.join(args.data_folder, args.dataset_name, 'wordmap', 'wordmap'+'.json')

  with open(word_map_file, 'r') as f:
    word_map = json.load(f)

  # Initialize
  if args.encoder == 'MCCFormers-D':
    encoder = MCCFormers_D(feature_dim = args.feature_dim, dropout=0.5,h=16,w=16,d_model=512,n_head=args.n_head,n_layers=args.n_layers).to(device)

  if args.encoder == 'MCCFormers-S':
    encoder = MCCFormers_S(feature_dim = args.feature_dim,h=16,w=16, n_head=args.n_head,n_layers=args.n_layers).to(device)

  if args.encoder == 'MyMCCFormers-S':
    encoder = MyMCCFormers_S(feature_dim = args.feature_dim,h=16,w=16, n_head=args.n_head,n_layers=args.n_layers).to(device)

  if args.encoder == 'AdvanceMCCFormers-S':
    encoder = AdvanceMCCFormers_S(feature_dim = args.feature_dim,h=16,w=16, n_head=args.n_head,n_layers=args.n_layers).to(device)

  if args.encoder == 'MyMCCFormers-D-1':
    encoder = MyMCCFormers_D_1(feature_dim = args.feature_dim,dropout=0.5,h=16,w=16,d_model=512,n_head=args.n_head,n_layers=args.n_layers).to(device)

  if args.encoder == 'MyMCCFormers-D-2':
    encoder = MyMCCFormers_D_2(feature_dim = args.feature_dim,dropout=0.5,h=16,w=16,d_model=512,n_head=args.n_head,n_layers=args.n_layers).to(device)
  
  if args.encoder == 'AdvanceMCCFormers-D':
    encoder = AdvanceMCCFormers_D(feature_dim = args.feature_dim,dropout=0.5,h=16,w=16,d_model=512,n_head=args.n_head,n_layers=args.n_layers).to(device)

  if args.decoder == 'trans':
    decoder = DecoderTransformer(feature_dim = args.feature_dim_de,
                               vocab_size = len(word_map),
                               n_head = args.n_head,
                               n_layers = args.n_layers,
                               dropout=dropout).to(device)


  if args.decoder == 'plain':
    decoder = PlainDecoder(feature_dim = args.feature_dim_de,
                           embed_dim = embed_dim,
                           vocab_size = len(word_map),
                           hidden_dim = args.hidden_dim,
                           dropout=dropout).to(device)
  
  encoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, encoder.parameters()),
                                       lr=encoder_lr)

  decoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, decoder.parameters()),
                                       lr=decoder_lr)

  criterion = nn.CrossEntropyLoss().to(device)

  train_loader = torch.utils.data.DataLoader(
    ModifiedCaptionDataset(args.data_folder, data_name, 'TRAIN', args.captions_per_image, args.dataset_name, encode_twice=args.encode_twice, wordmap=word_map, num_images=args.num_images, reverse_data=args.reverse_data),
    batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)

  for epoch in range(start_epoch, args.epochs):
    print("epoch : " + str(epoch))
    # print(f"?? {args.advance}")
    train(train_loader=train_loader,
          encoder=encoder,
          decoder=decoder,
          criterion=criterion,
          encoder_optimizer=encoder_optimizer,
          decoder_optimizer=decoder_optimizer,
          epoch=epoch,
          word_map=word_map,
          advance=args.advance,
          encode_twice=args.encode_twice
          )

    # Save checkpoint
    info = args.info
    info += "_encode_twice" if args.encode_twice else ""
    info += "_use_2_imgs" if args.advance else "_use_3_imgs" # or use args.num_images
    save_checkpoint(args.root_dir, args.encoder, args.dataset_name, epoch, encoder, decoder, encoder_optimizer, decoder_optimizer,info=info)
    

def train(train_loader, encoder, decoder, criterion, encoder_optimizer, decoder_optimizer, epoch, word_map, advance, encode_twice):
  encoder.train()
  decoder.train()

  batch_time = AverageMeter()
  data_time = AverageMeter()
  losses = AverageMeter()
  top3accs = AverageMeter()

  start = time.time()
  # print(f"Advance: {advance}")
  # Batches
  for i, data in enumerate(train_loader):
    # if advance:
    #   imgs1, imgs2, caps, caplens = data
    #   imgs3 = None
    # else:
    #   imgs1, imgs2, imgs3, caps, caplens = data
    caps2, caplens2, imgs3 = None, None, None
    if encode_twice:
      if advance:
        imgs1, imgs2, caps, caps2, caplens, caplens2 = data
      else:
        imgs1, imgs2, imgs3, caps, caps2, caplens, caplens2 = data
    else:
      if advance:
        imgs1, imgs2, caps, caplens = data
      else:
        imgs1, imgs2, imgs3, caps, caplens = data
    data_time.update(time.time() - start)
    # print("######")
    # print(caplens)
    # print(caps)
    # Move to GPU, if available
    imgs1 = imgs1.to(device)
    imgs2 = imgs2.to(device)
    if imgs3 is not None:
      imgs3 = imgs3.to(device)
    if caps2 is not None:
      caps2 = caps2.to(device)
    if caplens2 is not None:
      caplens2 = caplens2.to(device)

    caps = caps.to(device)
    caplens = caplens.to(device)
    
    # Forward prop.
    if advance:
      l = encoder(imgs1, imgs2, imgs3) # return memory
      scores, caps_sorted, decode_lengths, sort_ind = decoder(l, caps, caplens)
      targets = caps_sorted[:, 1:]

      scores = pack_padded_sequence(scores, decode_lengths, batch_first=True).data
      targets = pack_padded_sequence(targets, decode_lengths, batch_first=True).data
      loss = criterion(scores, targets)
          # Back prop.
      encoder_optimizer.zero_grad()
      decoder_optimizer.zero_grad()
      loss.backward()

      # Update weights
      encoder_optimizer.step()
      decoder_optimizer.step()

      # Keep track of metrics
      top3 = accuracy(scores, targets, 3)
      losses.update(loss.item(), sum(decode_lengths))
      top3accs.update(top3, sum(decode_lengths))
      batch_time.update(time.time() - start)
    else:
      if not isinstance(encoder, AdvanceMCCFormers_S) and not isinstance(encoder, AdvanceMCCFormers_D):
        l = encoder(imgs1, imgs2, imgs3)
        scores, caps_sorted, decode_lengths, sort_ind = decoder(l, caps, caplens)
        targets = caps_sorted[:, 1:]

        scores = pack_padded_sequence(scores, decode_lengths, batch_first=True).data
        targets = pack_padded_sequence(targets, decode_lengths, batch_first=True).data

        loss = criterion(scores, targets)
            # Back prop.
        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()
        loss.backward()

        # Update weights
        encoder_optimizer.step()
        decoder_optimizer.step()

        # Keep track of metrics
        top3 = accuracy(scores, targets, 3)
        losses.update(loss.item(), sum(decode_lengths))
        top3accs.update(top3, sum(decode_lengths))
        batch_time.update(time.time() - start)
      else:
        if encode_twice:
          l1, l2 = encoder(imgs1, imgs2, imgs3)

          scores_0, caps_sorted_0, decode_lengths_0, sort_ind = decoder(l1, caps, caplens)
          scores_1, caps_sorted_1, decode_lengths_1, sort_ind = decoder(l2, caps2, caplens2)
          # decode_lengths_1 = decode_lengths_2
          targets_0 = caps_sorted_0[:, 1:]
          targets_1 = caps_sorted_1[:, 1:]

          scores_0 = pack_padded_sequence(scores_0, decode_lengths_0, batch_first=True).data
          targets_0 = pack_padded_sequence(targets_0, decode_lengths_0, batch_first=True).data
          scores_1 = pack_padded_sequence(scores_1, decode_lengths_1, batch_first=True).data
          targets_1 = pack_padded_sequence(targets_1, decode_lengths_1, batch_first=True).data

          loss0 = criterion(scores_0, targets_0)
          loss1 = criterion(scores_1, targets_1)
          loss = loss0 + loss1

          # Back prop.
          encoder_optimizer.zero_grad()
          decoder_optimizer.zero_grad()
          loss.backward()

          # Update weights
          encoder_optimizer.step()
          decoder_optimizer.step()

          # Keep track of metrics
          top3 = accuracy(scores_0, targets_0, 3)
          top3_hat = accuracy(scores_1, targets_1, 3)
          losses.update(loss0.item(), sum(decode_lengths_0))
          losses.update(loss1.item(), sum(decode_lengths_1))
          top3accs.update(top3, sum(decode_lengths_0))
          top3accs.update(top3_hat, sum(decode_lengths_1))
          batch_time.update(time.time() - start)
        else:
          l1, l2 = encoder(imgs1, imgs2, imgs3)

          scores_0, caps_sorted_0, decode_lengths, sort_ind = decoder(l1, caps, caplens)
          scores_1, caps_sorted_1, decode_lengths, sort_ind = decoder(l2, caps, caplens)
          # decode_lengths_1 = decode_lengths_2
          targets_0 = caps_sorted_0[:, 1:]
          targets_1 = caps_sorted_1[:, 1:]

          scores_0 = pack_padded_sequence(scores_0, decode_lengths, batch_first=True).data
          targets_0 = pack_padded_sequence(targets_0, decode_lengths, batch_first=True).data
          scores_1 = pack_padded_sequence(scores_1, decode_lengths, batch_first=True).data
          targets_1 = pack_padded_sequence(targets_1, decode_lengths, batch_first=True).data
          scores = (scores_0 + scores_1)/2
          targets = targets_0

          loss = criterion(scores, targets)

          # Back prop.
          encoder_optimizer.zero_grad()
          decoder_optimizer.zero_grad()
          loss.backward()

          # Update weights
          encoder_optimizer.step()
          decoder_optimizer.step()

          # Keep track of metrics
          top3 = accuracy(scores, targets, 3)
          losses.update(loss.item(), sum(decode_lengths))
          top3accs.update(top3, sum(decode_lengths))
          batch_time.update(time.time() - start)
    
    



    start = time.time()
    
    # Print status
    if i % print_freq == 0:
      print('Epoch: [{0}][{1}/{2}]\t'
            'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
            'Data Load Time {data_time.val:.3f} ({data_time.avg:.3f})\t'
            'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
            'Top-3 Accuracy {top3.val:.3f} ({top3.avg:.3f})'.format(epoch, i, len(train_loader),
                                                                    batch_time=batch_time,
                                                                    data_time=data_time,
                                                                    loss=losses,
                                                                    top3=top3accs))

if __name__=='__main__':
  parser = argparse.ArgumentParser()

  parser.add_argument('--data_folder', default='dataset')
  parser.add_argument('--root_dir', default='result/')
  parser.add_argument('--hidden_dim', type=int, default=512)
  parser.add_argument('--attention_dim', type=int, default=512)
  parser.add_argument('--epochs', type=int, default=10)
  parser.add_argument('--encoder', default='MCCFormers-D')
  parser.add_argument('--decoder', default='trans')
  parser.add_argument('--n_head', type=int, default=4)
  parser.add_argument('--n_layers', type=int, default=2)
  parser.add_argument('--feature_dim', type=int, default=1024)
  parser.add_argument('--feature_dim_de', type=int, default=1024)
  parser.add_argument('--dataset_name', default='CLEVR')
  parser.add_argument('--advance', type=bool, default=False) # input any character meaning True!!!
  parser.add_argument('--encode_twice', type=bool, default=False)
  parser.add_argument('--info', default="", help="some mark of checkpoint ...")
  parser.add_argument('--captions_per_image', type=int, default=1)
  parser.add_argument('--num_images', type=int, default=3)
  parser.add_argument('--reverse_data', type=bool, default=False)

  args = parser.parse_args()

  # print(f"So {args.advance}")
  main(args)
  # dataset = ModifiedCaptionDataset(args.data_folder, data_name, 'TRAIN', args.captions_per_image, args.dataset_name, encode_twice=args.encode_twice, num_images=args.num_images)
  # print(len(dataset))
  # print(dataset[0])
