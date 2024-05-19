#!/bin/bash

# MyMCCFormers-S CLEVR no reverse
python modified_train_trans.py --data_folder dataset --encoder 'MyMCCFormers-S' --decoder trans --feature_dim_de 1536 --dataset_name 'CLEVR' --captions_per_image 1 --num_images 3 --info "first_noreverse" 

# MyMCCFormers-S CLEVR reverse
python modified_train_trans.py --data_folder dataset --encoder 'MyMCCFormers-S' --decoder trans --feature_dim_de 1536 --dataset_name 'CLEVR' --captions_per_image 1 --num_images 3 --info "first_reverse" --reverse_data 1

#MCCFormers-D-1 CLEVR no reverse
python modified_train_trans.py --data_folder dataset --encoder 'MyMCCFormers-D-1' --decoder trans --feature_dim_de 2048 --num_images 3 --captions_per_image 1 --dataset_name 'CLEVR' --info "first_noreverse" 

#MCCFormers-D-1 CLEVR reverse
python modified_train_trans.py --data_folder dataset --encoder 'MyMCCFormers-D-1' --decoder trans --feature_dim_de 2048 --num_images 3 --captions_per_image 1 --dataset_name 'CLEVR' --info "first_reverse" --reverse_data 1

#MCCFormers-D-2 CLEVR No reverse
python modified_train_trans.py --data_folder dataset --encoder 'MyMCCFormers-D-2' --decoder trans --feature_dim_de 1536 --num_images 3 --captions_per_image 1 --dataset_name "CLEVR" --info "first_noreverse"

#MCCFormers-D-2 CLEVR reverse
python modified_train_trans.py --data_folder dataset --encoder 'MyMCCFormers-D-2' --decoder trans --feature_dim_de 1536 --num_images 3 --captions_per_image 1 --dataset_name "CLEVR" --info "first_reverse" --reverse_data 1

#AdvanceFormers-S CLEVR no reverse
python modified_train_trans.py --data_folder dataset --encoder 'AdvanceMCCFormers-S' --decoder trans --feature_dim_de 1024 --num_images 3 --captions_per_image 1 --dataset_name "CLEVR" --info "first_base_noreverse"

#AdvanceFormers-S CLEVR reverse
python modified_train_trans.py --data_folder dataset --encoder 'AdvanceMCCFormers-S' --decoder trans --feature_dim_de 1024 --num_images 3 --captions_per_image 1 --dataset_name "CLEVR" --info "first_base_reverse" --reverse_data 1

#AdvanceFormers-S CLEVR no reverse, encode_twice
python modified_train_trans.py --data_folder dataset --encoder 'AdvanceMCCFormers-S' --decoder trans --feature_dim_de 1024 --num_images 3 --captions_per_image 1 --dataset_name "CLEVR" --info "first_base_noreverse_encode_twice" --encode_twice 1

#AdvanceFormers-S CLEVR reverse, encode_twice
python modified_train_trans.py --data_folder dataset --encoder 'AdvanceMCCFormers-S' --decoder trans --feature_dim_de 1024 --num_images 3 --captions_per_image 1 --dataset_name "CLEVR" --info "first_base_reverse_encode_twice" --reverse_data 1 --encode_twice 1

#AdvanceFormers-S CLEVRhat advance
python modified_train_trans.py --data_folder dataset --encoder 'AdvanceMCCFormers-S' --decoder trans --feature_dim_de 1024 --num_images 2 --captions_per_image 1 --dataset_name "CLEVRhat" --info "first_advance_CLEVRhat" --advance 1

#AdvanceFormers-D CLEVR base no reverse
python modified_train_trans.py --data_folder dataset --encoder 'AdvanceMCCFormers-D' --decoder trans --feature_dim_de 1024 --num_images 3 --captions_per_image 1 --dataset_name "CLEVR" --info "first_base_noreverse" 

#AdvanceFormers-D CLEVR base reverse
python modified_train_trans.py --data_folder dataset --encoder 'AdvanceMCCFormers-D' --decoder trans --feature_dim_de 1024 --num_images 3 --captions_per_image 1 --dataset_name "CLEVR" --info "first_base_reverse" --reverse_data 1

#AdvanceFormers-D CLEVR base no reverse, encoder_twice
python modified_train_trans.py --data_folder dataset --encoder 'AdvanceMCCFormers-D' --decoder trans --feature_dim_de 1024 --num_images 3 --captions_per_image 1 --dataset_name "CLEVR" --info "first_base_noreverse_encode_twice" --encode_twice 1

#AdvanceFormers-D CLEVR base reverse, encode_twice
python modified_train_trans.py --data_folder dataset --encoder 'AdvanceMCCFormers-D' --decoder trans --feature_dim_de 1024 --num_images 3 --captions_per_image 1 --dataset_name "CLEVR" --info "first_base_reverse_encode_twice" --reverse_data 1 --encode_twice 1

#AdvanceFormers-D CLEVRhat advance 
python modified_train_trans.py --data_folder dataset --encoder 'AdvanceMCCFormers-D' --decoder trans --feature_dim_de 1024 --num_images 2 --captions_per_image 1 --dataset_name "CLEVRhat" --info "first_advance_CLEVRhat" --advance 1

