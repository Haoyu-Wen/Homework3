import json
import os
import random

scenes_dir = '../output/scenes/'
output_dir = 'change_caption.json'

change_captions = []
change_captions_reverse = []
 
caption_templates = {
'add':['A <s> <c> <t> <z> has been added .',
       'A <s> <c> <t> <z> shows up .',
       'There is a new <s> <c> <t> <z> .',
       'A new <s> <c> <t> <z> is visible .',
       'Someone added a <s> <c> <t> <z> .'],
'delete':['The <s> <c> <t> <z> has disappeared .',
          'The <s> <c> <t> <z> is no longer there .',
          'The <s> <c> <t> <z> is missing .',
          'There is no longer a <s> <c> <t> <z> .',
          'Someone removed the <s> <c> <t> <z> .'],
'move':['The <s> <c> <t> <z> changed its location .',
        'The <s> <c> <t> <z> is in a different location .',
        'The <s> <c> <t> <z> was moved from its original location .',
        'The <s> <c> <t> <z> has been moved .',
        'Someone changed location of the <s> <c> <t> <z> .'],
'replace':['The <s> <c> <t> <z> was replaced by a <s1> <c1> <t1> <z1> .',
           'A <s1> <c1> <t1> <z1> replaced the <s> <c> <t> <z> .',
           'A <s1> <c1> <t1> <z1> is in the original position of <s> <c> <t> <z> .',
           'The <s> <c> <t> <z> gave up its position to a <s1> <c1> <t1> <z1> .',
           'Someone replaced the <s> <c> <t> <z> with a <s1> <c1> <t1> <z1> .']}

reverse_type={"add":"delete", "delete":"add", "replace":"replace", "move":"move"}

def instantiateCap(obj0, obj1, changetype):
  rand_sen_idx = random.randint(0,4)
  change_sentence = caption_templates[changetype][rand_sen_idx]
  
  change_sentence = change_sentence.replace('<s>',obj0['size']).replace('<c>',obj0['color']).replace('<t>',obj0['material']).replace('<z>',obj0['shape'])
  
  if changetype in ['add','delete','move']:
    return change_sentence
    
  change_sentence = change_sentence.replace('<s1>',obj1['size']).replace('<c1>',obj1['color']).replace('<t1>',obj1['material']).replace('<z1>',obj1['shape'])
  return change_sentence

def instantiateCapReverse(obj0, obj1, changetype):
  rand_sen_idx = random.randint(0, 4)
  change_sentence = caption_templates[reverse_type[changetype]][rand_sen_idx]
  
  if changetype in ['add', 'delete', 'move']:
    change_sentence = change_sentence.replace('<s>',obj0['size']).replace('<c>',obj0['color']).replace('<t>',obj0['material']).replace('<z>',obj0['shape'])
    return change_sentence
  
  change_sentence = change_sentence.replace('<s>',obj1['size']).replace('<c>',obj1['color']).replace('<t>',obj1['material']).replace('<z>',obj1['shape'])
  change_sentence = change_sentence.replace('<s1>',obj0['size']).replace('<c1>',obj0['color']).replace('<t1>',obj0['material']).replace('<z1>',obj0['shape'])
  return change_sentence

def getObj(change_type, order, current_info): 
  if change_type == 'add':
    return current_info['added_object'][order][0], None
    
  if change_type == 'delete':
    return current_info['dropped_object'][order], None    
  
  if change_type == 'move':
    return current_info['moved_object'][order][0], None    
    
  if change_type == 'replace':
    return current_info['replaced_object'][order], current_info['new_object'][order][0]         

def getOrder(change_record, idx):
  current_change = change_record[idx]
  order = 0 
  
  for i in range(0,idx):
    if change_record[i] == current_change:
      order = order + 1
      
  return order

def run():
  for scene in os.listdir(scenes_dir):
    with open(scenes_dir + scene, 'r') as f:
      current_info = json.load(f)
      
    curr_cap_info, curr_cap_info_reverse = {}, {}
    curr_cap_info['image_id'], curr_cap_info_reverse['image_id'] = current_info['image_filename'], current_info['image_filename']
    
    curr_cap_info['change_captions'], curr_cap_info_reverse['change_captions'] = [], []

    curr_cap_info['change_record'], curr_cap_info_reverse['change_record'] = current_info['change_record'], current_info['change_record']
    
    add_list = []
    delete_list = []
    move_list = []
    replace_list = []
    
    change_record = current_info['change_record']
    for idx in range(len(change_record)):
      order = getOrder(change_record, idx)
      obj0, obj1 = getObj(change_record[idx], order, current_info)
      curr_cap_info['change_captions'].append(instantiateCap(obj0, obj1, change_record[idx]))
      curr_cap_info_reverse['change_captions'].append(instantiateCapReverse(obj0, obj1, change_record[idx]))
    change_captions.append(curr_cap_info)
    change_captions_reverse.append(curr_cap_info_reverse)
    
  with open(output_dir, 'w') as f:
    json.dump(change_captions, f)

  with open("change_caption_reverse.json", 'w', encoding='utf-8') as f:
    json.dump(change_captions_reverse, f)

if __name__=="__main__":
  run()