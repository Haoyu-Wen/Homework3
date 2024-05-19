import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
from datasets import *
from utils import *
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu, SmoothingFunction
import torch.nn.functional as F
from tqdm import tqdm
import json

import argparse
import torch
# Parameters

data_name = '3dcc_5_cap_per_img_0_min_word_freq' # base name shared by data files

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # sets device for model and PyTorch tensors
cudnn.benchmark = True # set to true only if inputs to model are fixed size; otherwise lot of computational overhead

captions_per_image = 1
batch_size = 1

# model_name


def get_key(dict_, value):
  return [k for k, v in dict_.items() if v == value]

def beam_search_decoder(decoder, memory, word_map, rev_word_map, beam_size, max_len=120):
    k = beam_size
    vocab_size = len(word_map)
    
    start_token = word_map['<start>']
    end_token = word_map['<end>']
    
    # Initialize sequences with the start token
    sequences = [[start_token]]
    scores = torch.zeros(k, 1).to(device)
    
    for _ in range(max_len):
        all_candidates = []
        
        for i in range(len(sequences)):
            seq = sequences[i]
            score = scores[i]
            
            if seq[-1] == end_token:
                all_candidates.append((seq, score))
                continue
            
            tgt = torch.tensor(seq).unsqueeze(1).to(device)
            tgt_length = tgt.size(0)
            
            mask = (torch.triu(torch.ones(tgt_length, tgt_length)) == 1).transpose(0, 1)
            mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
            mask = mask.to(device)
            
            tgt_embedding = decoder.vocab_embedding(tgt)
            tgt_embedding = decoder.position_encoding(tgt_embedding)
            pred = decoder.transformer(tgt_embedding, memory, tgt_mask=mask)
            pred = decoder.wdc(pred)
            pred = pred[-1, 0, :]
            
            topk_probs, topk_indices = torch.topk(pred, k, dim=-1)
            
            for j in range(k):
                candidate = [seq + [topk_indices[j].item()], score - torch.log(topk_probs[j])]
                all_candidates.append(candidate)
        
        ordered = sorted(all_candidates, key=lambda x: x[1])
        sequences, scores = zip(*ordered[:k])
    
    best_sequence = sequences[0]
    return best_sequence

def top_k_sampling(decoder, memory, word_map, rev_word_map, k=5, max_len=80):
    start_token = word_map['<start>']
    end_token = word_map['<end>']
    
    # Initialize sequence with the start token
    seq = [start_token]
    
    for _ in range(max_len):
        tgt = torch.tensor(seq).unsqueeze(1).to(device)
        tgt_length = tgt.size(0)
        
        mask = (torch.triu(torch.ones(tgt_length, tgt_length)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        mask = mask.to(device)
        
        tgt_embedding = decoder.vocab_embedding(tgt)
        tgt_embedding = decoder.position_encoding(tgt_embedding)
        pred = decoder.transformer(tgt_embedding, memory, tgt_mask=mask)
        pred = decoder.wdc(pred)
        pred = pred[-1, 0, :]
        
        topk_probs, topk_indices = torch.topk(pred, k, dim=-1)
        
        # Normalize probabilities
        topk_probs = topk_probs / torch.sum(topk_probs)
        
        # Sample from the top k probabilities
        sampled_index = torch.multinomial(topk_probs, 1).item()
        next_word = topk_indices[sampled_index].item()
        
        if next_word == end_token:
            break
        
        seq.append(next_word)
    
    return seq

def evaluate(args, beam_size, n_gram):
  # Load model
  checkpoint = torch.load(args.checkpoint,map_location='cuda:0')

  encoder = checkpoint['encoder']
  encoder = encoder.to(device)
  encoder.eval()

  decoder = checkpoint['decoder']
  decoder = decoder.to(device)
  decoder.eval()

  word_map_file = os.path.join(args.data_folder, args.dataset_name, 'wordmap', 'wordmap.json')
  # Load word map (word2ix)
  with open(word_map_file, 'r') as f:
    word_map = json.load(f)

  rev_word_map = {v: k for k, v in word_map.items()}
  vocab_size = len(word_map)

  result_json_file = {}
  reference_json_file = {}


  """
  Evaluation

  :param beam_size: beam size at which to generate captions for evaluation
  :return: BLEU-4 score
  """

  # DataLoader
  loader = torch.utils.data.DataLoader(
      ModifiedCaptionDataset(args.data_folder, data_name, 'TEST', captions_per_image, args.dataset_name, num_images=3, wordmap=word_map),
      batch_size = batch_size, shuffle=False, num_workers=1, pin_memory=True)

  # TODO: Batched Beam Search
  # Therefore, do not use a batch_size greater than 1 - IMPORTANT!

  # Lists to store references (true captions), and hypothesis (prediction) for each image
  # If for n images, we have n hypotheses, and references a, b, c... for each image, we need -
  # references = [[ref1a, ref1b, ref1c], [ref2a, ref2b], ...], hypotheses = [hyp1, hyp2, ...]
  references = list()
  hypotheses = list()

  # For each image
  tmp = None
  ddd = 0
  for i, (image1, image2, image3, caps, caplens, allcaps) in enumerate(
    tqdm(loader, desc="EVALUATING AT BEAM SIZE " + str(beam_size))):

    # if (i % 5) != 0:
    #   continue

    current_index = i
    ddd += 1

    k = beam_size

    # Move to GPU device, if available
    image1 = image1.to(device) # 
    image2 = image2.to(device) #
    image3 = image3.to(device)

    memory = encoder(image1, image2, image3)
    if type(memory)==tuple:
      memory = (memory[0]+memory[1])/2
    # if tmp is not None:
    #   print("^^^^")
    #   print(tmp==memory)
    # print
    # tmp = memory
    # print(memory)
    
    seq = beam_search_decoder(decoder, memory, word_map, rev_word_map, beam_size)
    # seq = top_k_sampling(decoder, memory, word_map, rev_word_map, k=5, max_len=80)
    # tgt = torch.zeros(80,1).to(device).to(torch.int64)
    # tgt_length = tgt.size(0)

    # #print(tgt_length)

    # mask = (torch.triu(torch.ones(tgt_length, tgt_length)) == 1).transpose(0, 1)
    # mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    # mask = mask.to(device)

    # tgt[0,0] = word_map['<start>']
    # seq = []
    # for j in range(tgt_length-1):
    #   ##
    #   tgt_embedding = decoder.vocab_embedding(tgt) 
    #   tgt_embedding = decoder.position_encoding(tgt_embedding) #(length, batch, feature_dim)

    #   pred = decoder.transformer(tgt_embedding, memory, tgt_mask = mask) #(length, batch, feature_dim)
    #   pred = decoder.wdc(pred) #(length, batch, vocab_size)

    #   pred = pred[j,0,:]
    #   predicted_id = torch.argmax(pred, axis=-1)
   
    #   ## if word_map['<end>'], end for current sentence
    #   if predicted_id == word_map['<end>']:
    #     break

    #   seq.append(predicted_id)
    #   ## update mask, tgt
    #   tgt[j+1,0] = predicted_id
    #   mask[j+1,0] = 0.0


    # References
    
    # img_caps = allcaps[0].tolist()  ######################
    # print(img_caps)
    img_caps = caps.tolist()
    img_captions = list(
        map(lambda c: [w for w in c if w not in {word_map['<start>'], word_map['<end>'], word_map['<pad>']}],
            img_caps)) # remove <start> and pads
    references.append(img_captions)


    # Hypotheses
    temptemp = [w for w in seq if w not in {word_map['<start>'], word_map['<end>'], word_map['<pad>']}]
    hypotheses.append(temptemp)

    # assert len(references) == len(hypotheses)
    


  #-----------------------------------------------------------------
  kkk = -1
  for item in hypotheses:
    kkk += 1
    line_hypo = ""

    for word_idx in item:
      
      word = get_key(word_map, word_idx)
        #print(word)
      line_hypo += word[0] + " "

    result_json_file[str(kkk)] = []
    result_json_file[str(kkk)].append(line_hypo)

    line_hypo += "\r\n"


  kkk = -1
  for item in references:
    kkk += 1

    reference_json_file[str(kkk)] = []

    for sentence in item:
      line_repo = ""
      for word_idx in sentence:
        word = get_key(word_map, word_idx)
        line_repo += word[0] + " "
              
      reference_json_file[str(kkk)].append(line_repo)

      line_repo += "\r\n"

  result_root = os.path.join("eval_results", args.dataset_name, args.model_name)
  if not os.path.exists(result_root):
    os.makedirs(result_root, exist_ok=True)
  print(f"Saving file in {result_root+'/'+args.info+'xxx.json'}")
  with open(os.path.join(result_root, args.info+'res.json') ,'w') as f:
    json.dump(result_json_file,f)

  with open(os.path.join(result_root, args.info+'gts.json') ,'w') as f:
    json.dump(reference_json_file,f)

  assert len(result_json_file) == len(reference_json_file)
  blue_scores = []
  chencherry = SmoothingFunction()
  for (ref, result) in zip(reference_json_file, result_json_file):
    # blue_scores.append(sentence_bleu([ref.split()], result.split(), smoothing_function=chencherry.method1))
    blue_scores.append(sentence_bleu([ref.split()], result.split(), smoothing_function=chencherry.method1))
  avg_blue_score = sum(blue_scores)/len(blue_scores)
  print("############")
  print(f"Avg blue-score is {avg_blue_score:.3f}")

def r2(bleu):
  result = float(int(bleu*10000.0)/10000.0)
  
  result = str(result)
  while len(result) < 6:
    result += "0"

  return result


def evaluate_hat(args, beam_size, n_gram):
  # Load model
  checkpoint = torch.load(args.checkpoint,map_location='cuda:0')

  encoder = checkpoint['encoder']
  encoder = encoder.to(device)
  encoder.eval()

  decoder = checkpoint['decoder']
  decoder = decoder.to(device)
  decoder.eval()

  word_map_file = os.path.join(args.data_folder, args.dataset_name, 'wordmap', 'wordmap.json')
  # Load word map (word2ix)
  with open(word_map_file, 'r') as f:
    word_map = json.load(f)

  rev_word_map = {v: k for k, v in word_map.items()}
  vocab_size = len(word_map)

  result_json_file = {}
  reference_json_file = {}


  """
  Evaluation

  :param beam_size: beam size at which to generate captions for evaluation
  :return: BLEU-4 score
  """

  # DataLoader
  loader = torch.utils.data.DataLoader(
      ModifiedCaptionDataset(args.data_folder, data_name, 'TEST', captions_per_image,args.dataset_name, encode_twice=True, wordmap=word_map, num_images=3),
      batch_size = batch_size, shuffle=False, num_workers=1, pin_memory=True)
  
  # TODO: Batched Beam Search
  # Therefore, do not use a batch_size greater than 1 - IMPORTANT!

  # Lists to store references (true captions), and hypothesis (prediction) for each image
  # If for n images, we have n hypotheses, and references a, b, c... for each image, we need -
  # references = [[ref1a, ref1b, ref1c], [ref2a, ref2b], ...], hypotheses = [hyp1, hyp2, ...]
  references = list()
  hypotheses = list()

  # For each image
  ddd = 0
  for i, (image1, image2, image3, caps1, caps2, caplens1, caplens2, allcaps1, allcaps2) in enumerate(
          tqdm(loader, desc="EVALUATING AT BEAM SIZE " + str(beam_size))):

    # if (i % 5) != 0:
    #   continue

    current_index = i
    ddd += 1

    k = beam_size

    # Move to GPU device, if available
    image1 = image1.to(device) # 
    image2 = image2.to(device) #
    image3 = image3.to(device)

    memory1, memory2 = encoder(image1, image2, image3)
    
    # tgt1 = torch.zeros(80,1).to(device).to(torch.int64)
    # tgt2 = torch.zeros(80,1).to(device).to(torch.int64)
    # tgt_length = tgt1.size(0)

    # #print(tgt_length)

    # mask1 = (torch.triu(torch.ones(tgt_length, tgt_length)) == 1).transpose(0, 1)
    # mask1 = mask1.float().masked_fill(mask1 == 0, float('-inf')).masked_fill(mask1 == 1, float(0.0))
    # mask1 = mask1.to(device)
    # mask2 = (torch.triu(torch.ones(tgt_length, tgt_length)) == 1).transpose(0, 1)
    # mask2 = mask2.float().masked_fill(mask2 == 0, float('-inf')).masked_fill(mask2 == 1, float(0.0))
    # mask2 = mask2.to(device)

    # tgt1[0,0] = word_map['<start>']
    # tgt2[0,0] = word_map['<start>']
    seq1, seq2 = [], []
    seq1 = beam_search_decoder(decoder, memory1, word_map, rev_word_map, beam_size)
    seq2 = beam_search_decoder(decoder, memory2, word_map, rev_word_map, beam_size)
    # description between img1 and img2
    # seq1, tgt1...
    # for i in range(tgt_length-1):
    #   ##
    #   tgt_embedding = decoder.vocab_embedding(tgt1) 
    #   tgt_embedding = decoder.position_encoding(tgt_embedding) #(length, batch, feature_dim)

    #   pred1 = decoder.transformer(tgt_embedding, memory1, tgt_mask = mask1) #(length, batch, feature_dim)
    #   pred1 = decoder.wdc(pred1) #(length, batch, vocab_size)

    #   pred1 = pred1[i,0,:]
    #   predicted_id1 = torch.argmax(pred1, axis=-1)
   
    #   ## if word_map['<end>'], end for current sentence
    #   if predicted_id1 == word_map['<end>']:
    #     break

    #   seq1.append(predicted_id1)

    #   ## update mask, tgt
    #   tgt1[i+1,0] = predicted_id1
    #   mask1[i+1,0] = 0.0


    # # description between img2 and img3
    # # seq2, tgt2
    # for i in range(tgt_length-1):
    #   ##
    #   tgt_embedding = decoder.vocab_embedding(tgt2) 
    #   tgt_embedding = decoder.position_encoding(tgt_embedding) #(length, batch, feature_dim)

    #   pred2 = decoder.transformer(tgt_embedding, memory2, tgt_mask = mask2) #(length, batch, feature_dim)
    #   pred2 = decoder.wdc(pred2) #(length, batch, vocab_size)

    #   pred2 = pred2[i,0,:]
    #   predicted_id2 = torch.argmax(pred2, axis=-1)
   
    #   ## if word_map['<end>'], end for current sentence
    #   if predicted_id2 == word_map['<end>']:
    #     break

    #   seq2.append(predicted_id2)

    #   ## update mask, tgt
    #   tgt2[i+1,0] = predicted_id2
    #   mask2[i+1,0] = 0.0


    # References
    img_caps1 = caps1.tolist()  ######################
    img_caps2 = caps2.tolist()
    img_captions1 = list(
        map(lambda c: [w for w in c if w not in {word_map['<start>'], word_map['<end>'], word_map['<pad>']}],
            img_caps1)) # remove <start> and pads
    img_captions2 = list(
        map(lambda c: [w for w in c if w not in {word_map['<start>'], word_map['<end>'], word_map['<pad>']}],
            img_caps2))
    img_captions1.extend(img_captions2)
    references.append(img_captions1)


    # Hypotheses
    temptemp = [w for w in seq1 if w not in {word_map['<start>'], word_map['<end>'], word_map['<pad>']}]
    temptemp.extend([w for w in seq2 if w not in {word_map['<start>'], word_map['<end>'], word_map['<pad>']}])
    hypotheses.append(temptemp)

    assert len(references) == len(hypotheses)


  #-----------------------------------------------------------------
  kkk = -1
  for item in hypotheses:
    kkk += 1
    line_hypo = ""

    for word_idx in item:
      word = get_key(word_map, word_idx)
        #print(word)
      line_hypo += word[0] + " "

    result_json_file[str(kkk)] = []
    result_json_file[str(kkk)].append(line_hypo)

    line_hypo += "\r\n"


  kkk = -1
  for item in references:
    kkk += 1

    reference_json_file[str(kkk)] = []

    for sentence in item:
      line_repo = ""
      for word_idx in sentence:
        word = get_key(word_map, word_idx)
        line_repo += word[0] + " "
              
      reference_json_file[str(kkk)].append(line_repo)

      line_repo += "\r\n"

  result_root = os.path.join("eval_results", args.dataset_name, args.model_name)
  if not os.path.exists(result_root):
    os.makedirs(result_root, exist_ok=True)
  print(f"Saving eval result in {result_root}/{args.info}_decode_twice_xxx.json")
  with open(os.path.join(result_root, args.info+'decode_twice_res.json') ,'w') as f:
    json.dump(result_json_file,f)

  with open(os.path.join(result_root, args.info+'decode_twice_gts.json') ,'w') as f:
    json.dump(reference_json_file,f)

  assert len(result_json_file) == len(reference_json_file)
  blue_scores = []
  chencherry = SmoothingFunction()
  for (ref, result) in zip(reference_json_file, result_json_file):
    blue_scores.append(sentence_bleu([ref.split()], result.split(), smoothing_function=chencherry.method1))
  avg_blue_score = sum(blue_scores)/len(blue_scores)
  print("############")
  print(f"Avg blue-score is {avg_blue_score:.3f}")



if __name__=='__main__':
  parser = argparse.ArgumentParser()
  

  parser.add_argument('--data_folder', default='dataset')
  parser.add_argument('--checkpoint', default='result/checkpoint_epoch_39_3dcc_5_cap_per_img_0_min_word_freq.pth.tar')
  # parser.add_argument('--word_map_file', default='dataset/wordmap/wordmap.json')
  parser.add_argument('--model_name', default='con-sub_too_r3')
  parser.add_argument('--dataset_name', default='CLEVR')
  parser.add_argument('--decode_twice', type=bool, default=False)
  parser.add_argument('--info', default="", help="custom info for saving result json file name")
  parser.add_argument("--beam_size", type=int, default=3)

  args = parser.parse_args()

  beam_size = args.beam_size
  n_gram = 4

  if args.decode_twice:
    evaluate_hat(args, beam_size, n_gram)
  else:
    evaluate(args, beam_size, n_gram)










































