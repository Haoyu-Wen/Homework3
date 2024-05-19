#!/bin/bash
# should cd ./code
#checkpoint path and dataset_name should pay attention!!!
#如果按照下面的脚本将打印的内容重定向到某个txt，需要提前在code文件夹下创建eval_results/infos文件夹

# MCCFormers-S base no reverse
#epoch 4, beam=3
python modified_eval_trans.py --data_folder dataset --dataset_name 'CLEVR' --beam_size 3 --checkpoint result/MyMCCFormers-S/CLEVR/checkpoint_epoch_4first_noreverse_use_3_imgs.pth.tar --model_name 'MyMCCFormers-S' --info 'base_no_reverse_epoch4' >eval_results/infos/MyMCCFomers_S_no_reverse_beamsize3_epoch4.txt
#epoch 9, beam =3
python modified_eval_trans.py --data_folder dataset --dataset_name 'CLEVR' --beam_size 3 --checkpoint result/MyMCCFormers-S/CLEVR/checkpoint_epoch_9first_noreverse_use_3_imgs.pth.tar --model_name 'MyMCCFormers-S' --info 'base_no_reverse_epoch9' >eval_results/infos/MyMCCFomers_S_no_reverse_beamsize3_epoch9.txt

#MCCFormers-S base reverse
#epoch 4, beam=3
python modified_eval_trans.py --data_folder dataset --dataset_name 'CLEVR' --beam_size 3 --checkpoint result/MyMCCFormers-S/CLEVR/checkpoint_epoch_4first_reverse_use_3_imgs.pth.tar --model_name 'MyMCCFormers-S' --info 'base_reverse_epoch4' >eval_results/infos/MyMCCFomers_S_reverse_beamsize3_epoch4.txt
#epoch 9, beam=3
python modified_eval_trans.py --data_folder dataset --dataset_name 'CLEVR' --beam_size 3 --checkpoint result/MyMCCFormers-S/CLEVR/checkpoint_epoch_9first_reverse_use_3_imgs.pth.tar --model_name 'MyMCCFormers-S' --info 'base_reverse_epoch9' >eval_results/infos/MyMCCFomers_S_reverse_beamsize3_epoch9.txt

#MCCFormers-D1 base 
#no reverse
#base no reverse, epoch 4, beam=3
python modified_eval_trans.py --data_folder dataset --dataset_name 'CLEVR' --beam_size 3 --checkpoint result/MyMCCFormers-D-1/CLEVR/checkpoint_epoch_4first_noreverse_use_3_imgs.pth.tar --model_name 'MyMCCFormers-D-1' --info 'base_no_reverse_epoch4' >eval_results/infos/MyMCCFomers_D1_no_reverse_beamsize3_epoch4.txt
#base no reverse, epoch 9, beam=3
python modified_eval_trans.py --data_folder dataset --dataset_name 'CLEVR' --beam_size 3 --checkpoint result/MyMCCFormers-D-1/CLEVR/checkpoint_epoch_9first_noreverse_use_3_imgs.pth.tar --model_name 'MyMCCFormers-D-1' --info 'base_no_reverse_epoch9' >eval_results/infos/MyMCCFomers_D1_no_reverse_beamsize3_epoch9.txt

#reverse
#base, reverse, epoch=4, beam=3
python modified_eval_trans.py --data_folder dataset --dataset_name 'CLEVR' --beam_size 3 --checkpoint result/MyMCCFormers-D-1/CLEVR/checkpoint_epoch_4first_reverse_use_3_imgs.pth.tar --model_name 'MyMCCFormers-D-1' --info 'base_reverse_epoch4' >eval_results/infos/MyMCCFomers_D1_reverse_beamsize3_epoch4.txt
#base, reverse, epoch=9, beam=3
python modified_eval_trans.py --data_folder dataset --dataset_name 'CLEVR' --beam_size 3 --checkpoint result/MyMCCFormers-D-1/CLEVR/checkpoint_epoch_9first_reverse_use_3_imgs.pth.tar --model_name 'MyMCCFormers-D-1' --info 'base_reverse_epoch9' >eval_results/infos/MyMCCFomers_D1_reverse_beamsize3_epoch9.txt
#MCCFormers-D2 base
#no reverse
#base, no reverse, epocg=4, beam=3
python modified_eval_trans.py --data_folder dataset --dataset_name 'CLEVR' --beam_size 3 --checkpoint result/MyMCCFormers-D-2/CLEVR/checkpoint_epoch_4first_no_reverse_use_3_imgs.pth.tar --model_name 'MyMCCFormers-D-2' --info 'base_noreverse_epoch4' >eval_results/infos/MyMCCFormers_D2_no_reverse_beamsize3_epoch4.txt
#base, no reverse, epocg=9, beam=3
python modified_eval_trans.py --data_folder dataset --dataset_name 'CLEVR' --beam_size 3 --checkpoint result/MyMCCFormers-D-2/CLEVR/checkpoint_epoch_9first_no_reverse_use_3_imgs.pth.tar --model_name 'MyMCCFormers-D-2' --info 'base_noreverse_epoch9' >eval_results/infos/MyMCCFormers_D2_no_reverse_beamsize3_epoch9.txt
# reverse
#base,  reverse, epocg=4, beam=3
python modified_eval_trans.py --data_folder dataset --dataset_name 'CLEVR' --beam_size 3 --checkpoint result/MyMCCFormers-D-2/CLEVR/checkpoint_epoch_4first_reverse_use_3_imgs.pth.tar --model_name 'MyMCCFormers-D-2' --info 'base_reverse_epoch4' >eval_results/infos/MyMCCFormers_D2_reverse_beamsize3_epoch4.txt
#base,  reverse, epocg=9, beam=3
python modified_eval_trans.py --data_folder dataset --dataset_name 'CLEVR' --beam_size 3 --checkpoint result/MyMCCFormers-D-2/CLEVR/checkpoint_epoch_9first_reverse_use_3_imgs.pth.tar --model_name 'MyMCCFormers-D-2' --info 'base_reverse_epoch9' >eval_results/infos/MyMCCFormers_D2_reverse_beamsize3_epoch9.txt
#AdvanceMCCFormers-S
# base, CLEVR, no reverse
#epoch=4, beam=3
python modified_eval_trans.py --data_folder dataset --dataset_name 'CLEVR' --beam_size 3 --checkpoint result/AdvanceMCCFormers-S/CLEVR/checkpoint_epoch_4first_base_noreverse_use_3_imgs.pth.tar --model_name 'AdvanceMCCFormers-S' --info 'base_no_reverse_CLEVR_epoch4' >eval_results/infos/AdvanceMCCFormers_S_CLEVR_no_reverse_beamsize_3_epoch4.txt
#epoch=9, beam=3
python modified_eval_trans.py --data_folder dataset --dataset_name 'CLEVR' --beam_size 3 --checkpoint result/AdvanceMCCFormers-S/CLEVR/checkpoint_epoch_9first_base_noreverse_use_3_imgs.pth.tar --model_name 'AdvanceMCCFormers-S' --info 'base_no_reverse_CLEVR_epoch9' >eval_results/infos/AdvanceMCCFormers_S_CLEVR_no_reverse_beamsize_3_epoch9.txt
# decoder_twice
#epoch=4, beam=3
python modified_eval_trans.py --data_folder dataset --dataset_name 'CLEVR' --beam_size 3 --checkpoint result/AdvanceMCCFormers-S/CLEVR/checkpoint_epoch_4first_base_noreverse_use_3_imgs.pth.tar --model_name 'AdvanceMCCFormers-S' --info 'base_no_reverse_CLEVR_decode_twice_epoch4' --decode_twice 1 >eval_results/infos/AdvanceMCCFormers_S_CLEVR_no_reverse_beamsize_3_decode_twice_epoch4.txt
#epoch=9, beam=3
python modified_eval_trans.py --data_folder dataset --dataset_name 'CLEVR' --beam_size 3 --checkpoint result/AdvanceMCCFormers-S/CLEVR/checkpoint_epoch_9first_base_noreverse_use_3_imgs.pth.tar --model_name 'AdvanceMCCFormers-S' --info 'base_no_reverse_CLEVR_decode_twice_epoch9' --decode_twice 1 >eval_results/infos/AdvanceMCCFormers_S_CLEVR_no_reverse_beamsize_3_decode_twice_epoch9.txt

#base, CLEVR, reverse
#epoch=4, beam=3
python modified_eval_trans.py --data_folder dataset --dataset_name 'CLEVR' --beam_size 3 --checkpoint result/AdvanceMCCFormers-S/CLEVR/checkpoint_epoch_4first_base_reverse_use_3_imgs.pth.tar --model_name 'AdvanceMCCFormers-S' --info 'base_reverse_CLEVR_epoch4' >eval_results/infos/AdvanceMCCFormers_S_CLEVR_reverse_beamsize_3_epoch4.txt
#epoch=9, beam=3
python modified_eval_trans.py --data_folder dataset --dataset_name 'CLEVR' --beam_size 3 --checkpoint result/AdvanceMCCFormers-S/CLEVR/checkpoint_epoch_9first_base_reverse_use_3_imgs.pth.tar --model_name 'AdvanceMCCFormers-S' --info 'base_reverse_CLEVR_epoch9' >eval_results/infos/AdvanceMCCFormers_S_CLEVR_reverse_beamsize_3_epoch9.txt
# decoder_twice
#epoch=4, beam=3
python modified_eval_trans.py --data_folder dataset --dataset_name 'CLEVR' --beam_size 3 --checkpoint result/AdvanceMCCFormers-S/CLEVR/checkpoint_epoch_4first_base_reverse_use_3_imgs.pth.tar --model_name 'AdvanceMCCFormers-S' --info 'base_reverse_CLEVR_decode_twice_epoch4' --decode_twice 1 >eval_results/infos/AdvanceMCCFormers_S_CLEVR_reverse_beamsize_3_decode_twice_epoch4.txt
#epoch=9, beam=3
python modified_eval_trans.py --data_folder dataset --dataset_name 'CLEVR' --beam_size 3 --checkpoint result/AdvanceMCCFormers-S/CLEVR/checkpoint_epoch_9first_base_reverse_use_3_imgs.pth.tar --model_name 'AdvanceMCCFormers-S' --info 'base_reverse_CLEVR_decode_twice_epoch9' --decode_twice 1 >eval_results/infos/AdvanceMCCFormers_S_CLEVR_reverse_beamsize_3_decode_twice_epoch9.txt

####encode_twice
# base, CLEVR, no reverse
#epoch=4, beam=3
python modified_eval_trans.py --data_folder dataset --dataset_name 'CLEVR' --beam_size 3 --checkpoint result/AdvanceMCCFormers-S/CLEVR/checkpoint_epoch_4first_base_noreverse_encode_twice_encode_twice_use_3_imgs.pth.tar --model_name 'AdvanceMCCFormers-S' --info 'base_no_reverse_CLEVR_epoch4_encode_twice' >eval_results/infos/AdvanceMCCFormers_S_CLEVR_no_reverse_beamsize_3_encode_twice_epoch4.txt
#epoch=9, beam=3
python modified_eval_trans.py --data_folder dataset --dataset_name 'CLEVR' --beam_size 3 --checkpoint result/AdvanceMCCFormers-S/CLEVR/checkpoint_epoch_9first_base_noreverse_encode_twice_encode_twice_use_3_imgs.pth.tar --model_name 'AdvanceMCCFormers-S' --info 'base_no_reverse_CLEVR_epoch9_encode_twice' >eval_results/infos/AdvanceMCCFormers_S_CLEVR_no_reverse_beamsize_3_encode_twice_epoch9.txt
# decoder_twice
#epoch=4, beam=3
python modified_eval_trans.py --data_folder dataset --dataset_name 'CLEVR' --beam_size 3 --checkpoint result/AdvanceMCCFormers-S/CLEVR/checkpoint_epoch_4first_base_noreverse_encode_twice_encode_twice_use_3_imgs.pth.tar --model_name 'AdvanceMCCFormers-S' --info 'base_no_reverse_CLEVR_decode_twice_epoch4_encode_twice' --decode_twice 1 >eval_results/infos/AdvanceMCCFormers_S_CLEVR_no_reverse_beamsize_3_encode_twice_decode_twice_epoch4.txt
#epoch=9, beam=3
python modified_eval_trans.py --data_folder dataset --dataset_name 'CLEVR' --beam_size 3 --checkpoint result/AdvanceMCCFormers-S/CLEVR/checkpoint_epoch_9first_base_noreverse_encode_twice_encode_twice_use_3_imgs.pth.tar --model_name 'AdvanceMCCFormers-S' --info 'base_no_reverse_CLEVR_decode_twice_epoch9_encode_twice' --decode_twice 1 >eval_results/infos/AdvanceMCCFormers_S_CLEVR_no_reverse_beamsize_3_encode_twice_decode_twice_epoch9.txt

#base, CLEVR, reverse
#epoch=4, beam=3
python modified_eval_trans.py --data_folder dataset --dataset_name 'CLEVR' --beam_size 3 --checkpoint result/AdvanceMCCFormers-S/CLEVR/checkpoint_epoch_4first_base_reverse_encode_twice_encode_twice_use_3_imgs.pth.tar --model_name 'AdvanceMCCFormers-S' --info 'base_reverse_CLEVR_epoch4_encode_twice' >eval_results/infos/AdvanceMCCFormers_S_CLEVR_reverse_beamsize_3_encode_twice_epoch4.txt
#epoch=9, beam=3
python modified_eval_trans.py --data_folder dataset --dataset_name 'CLEVR' --beam_size 3 --checkpoint result/AdvanceMCCFormers-S/CLEVR/checkpoint_epoch_9first_base_reverse_encode_twice_encode_twice_use_3_imgs.pth.tar --model_name 'AdvanceMCCFormers-S' --info 'base_reverse_CLEVR_epoch9_encode_twice' >eval_results/infos/AdvanceMCCFormers_S_CLEVR_reverse_beamsize_3_encode_twice_epoch9.txt
# decoder_twice
#epoch=4, beam=3
python modified_eval_trans.py --data_folder dataset --dataset_name 'CLEVR' --beam_size 3 --checkpoint result/AdvanceMCCFormers-S/CLEVR/checkpoint_epoch_4first_base_reverse_encode_twice_encode_twice_use_3_imgs.pth.tar --model_name 'AdvanceMCCFormers-S' --info 'base_reverse_CLEVR_decode_twice_epoch4_encode_twice' --decode_twice 1 >eval_results/infos/AdvanceMCCFormers_S_CLEVR_reverse_beamsize_3_encode_twice_decode_twice_epoch4.txt
#epoch=9, beam=3
python modified_eval_trans.py --data_folder dataset --dataset_name 'CLEVR' --beam_size 3 --checkpoint result/AdvanceMCCFormers-S/CLEVR/checkpoint_epoch_9first_base_reverse_encode_twice_encode_twice_use_3_imgs.pth.tar --model_name 'AdvanceMCCFormers-S' --info 'base_reverse_CLEVR_decode_twice_epoch9_encode_twice' --decode_twice 1 >eval_results/infos/AdvanceMCCFormers_S_CLEVR_reverse_beamsize_3_encode_twice_decode_twice_epoch9.txt

# advance, train CLEVRhat, test CLEVR
#only no reverse
#epoch 4
python modified_eval_trans.py --data_folder dataset --dataset_name 'CLEVR' --beam_size 3 --checkpoint result/AdvanceMCCFormers-S/CLEVRhat/checkpoint_epoch_4first_advance_CLEVRhat_use_2_imgs.pth.tar --model_name 'AdvanceMCCFormers-S' --info 'advance_CLEVRhat_epoch4' > eval_results/infos/AdvanceMCCFormers_S_CLEVR_beamsize_3_epoch4.txt
#epoch 9
python modified_eval_trans.py --data_folder dataset --dataset_name 'CLEVR' --beam_size 3 --checkpoint result/AdvanceMCCFormers-S/CLEVRhat/checkpoint_epoch_9first_advance_CLEVRhat_use_2_imgs.pth.tar --model_name 'AdvanceMCCFormers-S' --info 'advance_CLEVRhat_epoch9' > eval_results/infos/AdvanceMCCFormers_S_CLEVR_beamsize_3_epoch9.txt
#decode twice
python modified_eval_trans.py --data_folder dataset --dataset_name 'CLEVR' --beam_size 3 --checkpoint result/AdvanceMCCFormers-S/CLEVRhat/checkpoint_epoch_4first_advance_CLEVRhat_use_2_imgs.pth.tar --model_name 'AdvanceMCCFormers-S' --info 'advance_CLEVRhat_decode_twice_epoch4' --decode_twice 1 > eval_results/infos/AdvanceMCCFormers_S_CLEVR_beamsize_3_decode_twice_epoch4.txt
#epoch 9
python modified_eval_trans.py --data_folder dataset --dataset_name 'CLEVR' --beam_size 3 --checkpoint result/AdvanceMCCFormers-S/CLEVRhat/checkpoint_epoch_9first_advance_CLEVRhat_use_2_imgs.pth.tar --model_name 'AdvanceMCCFormers-S' --info 'advance_CLEVRhat_decode_twice_epoch9' --decode_twice 1 > eval_results/infos/AdvanceMCCFormers_S_CLEVR_beamsize_3_decode_twice_epoch9.txt


#AdvanceMCCFormers-D
# base, CLEVR, no reverse
#epoch=4, beam=3
python modified_eval_trans.py --data_folder dataset --dataset_name 'CLEVR' --beam_size 3 --checkpoint result/AdvanceMCCFormers-D/CLEVR/checkpoint_epoch_4first_base_noreverse_use_3_imgs.pth.tar --model_name 'AdvanceMCCFormers-D' --info 'base_no_reverse_CLEVR_epoch4' >eval_results/infos/AdvanceMCCFormers_D_CLEVR_no_reverse_beamsize_3_epoch4.txt
#epoch=9, beam=3
python modified_eval_trans.py --data_folder dataset --dataset_name 'CLEVR' --beam_size 3 --checkpoint result/AdvanceMCCFormers-D/CLEVR/checkpoint_epoch_9first_base_noreverse_use_3_imgs.pth.tar --model_name 'AdvanceMCCFormers-D' --info 'base_no_reverse_CLEVR_epoch9' >eval_results/infos/AdvanceMCCFormers_D_CLEVR_no_reverse_beamsize_3_epoch9.txt
# decoder_twice
#epoch=4, beam=3
python modified_eval_trans.py --data_folder dataset --dataset_name 'CLEVR' --beam_size 3 --checkpoint result/AdvanceMCCFormers-D/CLEVR/checkpoint_epoch_4first_base_noreverse_use_3_imgs.pth.tar --model_name 'AdvanceMCCFormers-D' --info 'base_no_reverse_CLEVR_decode_twice_epoch4' --decode_twice 1 >eval_results/infos/AdvanceMCCFormers_D_CLEVR_no_reverse_beamsize_3_decode_twice_epoch4.txt
#epoch=9, beam=3
python modified_eval_trans.py --data_folder dataset --dataset_name 'CLEVR' --beam_size 3 --checkpoint result/AdvanceMCCFormers-D/CLEVR/checkpoint_epoch_9first_base_noreverse_use_3_imgs.pth.tar --model_name 'AdvanceMCCFormers-D' --info 'base_no_reverse_CLEVR_decode_twice_epoch9' --decode_twice 1 >eval_results/infos/AdvanceMCCFormers_D_CLEVR_no_reverse_beamsize_3_decode_twice_epoch9.txt

#base, CLEVR, reverse
#epoch=4, beam=3
python modified_eval_trans.py --data_folder dataset --dataset_name 'CLEVR' --beam_size 3 --checkpoint result/AdvanceMCCFormers-D/CLEVR/checkpoint_epoch_4first_base_reverse_use_3_imgs.pth.tar --model_name 'AdvanceMCCFormers-D' --info 'base_reverse_CLEVR_epoch4' >eval_results/infos/AdvanceMCCFormers_D_CLEVR_reverse_beamsize_3_epoch4.txt
#epoch=9, beam=3
python modified_eval_trans.py --data_folder dataset --dataset_name 'CLEVR' --beam_size 3 --checkpoint result/AdvanceMCCFormers-D/CLEVR/checkpoint_epoch_9first_base_reverse_use_3_imgs.pth.tar --model_name 'AdvanceMCCFormers-D' --info 'base_reverse_CLEVR_epoch9' >eval_results/infos/AdvanceMCCFormers_D_CLEVR_reverse_beamsize_3_epoch9.txt
# decoder_twice
#epoch=4, beam=3
python modified_eval_trans.py --data_folder dataset --dataset_name 'CLEVR' --beam_size 3 --checkpoint result/AdvanceMCCFormers-D/CLEVR/checkpoint_epoch_4first_base_reverse_use_3_imgs.pth.tar --model_name 'AdvanceMCCFormers-D' --info 'base_reverse_CLEVR_decode_twice_epoch4' --decode_twice 1 >eval_results/infos/AdvanceMCCFormers_D_CLEVR_reverse_beamsize_3_decode_twice_epoch4.txt
#epoch=9, beam=3
python modified_eval_trans.py --data_folder dataset --dataset_name 'CLEVR' --beam_size 3 --checkpoint result/AdvanceMCCFormers-D/CLEVR/checkpoint_epoch_9first_base_reverse_use_3_imgs.pth.tar --model_name 'AdvanceMCCFormers-D' --info 'base_reverse_CLEVR_decode_twice_epoch9' --decode_twice 1 >eval_results/infos/AdvanceMCCFormers_D_CLEVR_reverse_beamsize_3_decode_twice_epoch9.txt

####encode_twice
# base, CLEVR, no reverse
#epoch=4, beam=3
python modified_eval_trans.py --data_folder dataset --dataset_name 'CLEVR' --beam_size 3 --checkpoint result/AdvanceMCCFormers-D/CLEVR/checkpoint_epoch_4first_base_noreverse_encode_twice_encode_twice_use_3_imgs.pth.tar --model_name 'AdvanceMCCFormers-D' --info 'base_no_reverse_CLEVR_epoch4_encode_twice' >eval_results/infos/AdvanceMCCFormers_D_CLEVR_no_reverse_beamsize_3_encode_twice_epoch4.txt
#epoch=9, beam=3
python modified_eval_trans.py --data_folder dataset --dataset_name 'CLEVR' --beam_size 3 --checkpoint result/AdvanceMCCFormers-D/CLEVR/checkpoint_epoch_9first_base_noreverse_encode_twice_encode_twice_use_3_imgs.pth.tar --model_name 'AdvanceMCCFormers-D' --info 'base_no_reverse_CLEVR_epoch9_encode_twice' >eval_results/infos/AdvanceMCCFormers_D_CLEVR_no_reverse_beamsize_3_encode_twice_epoch9.txt
# decoder_twice
#epoch=4, beam=3
python modified_eval_trans.py --data_folder dataset --dataset_name 'CLEVR' --beam_size 3 --checkpoint result/AdvanceMCCFormers-D/CLEVR/checkpoint_epoch_4first_base_noreverse_encode_twice_encode_twice_use_3_imgs.pth.tar --model_name 'AdvanceMCCFormers-D' --info 'base_no_reverse_CLEVR_decode_twice_epoch4_encode_twice' --decode_twice 1 >eval_results/infos/AdvanceMCCFormers_D_CLEVR_no_reverse_beamsize_3_encode_twice_decode_twice_epoch4.txt
#epoch=9, beam=3
python modified_eval_trans.py --data_folder dataset --dataset_name 'CLEVR' --beam_size 3 --checkpoint result/AdvanceMCCFormers-D/CLEVR/checkpoint_epoch_9first_base_noreverse_encode_twice_encode_twice_use_3_imgs.pth.tar --model_name 'AdvanceMCCFormers-D' --info 'base_no_reverse_CLEVR_decode_twice_epoch9_encode_twice' --decode_twice 1 >eval_results/infos/AdvanceMCCFormers_D_CLEVR_no_reverse_beamsize_3_encode_twice_decode_twice_epoch9.txt

#base, CLEVR, reverse
#epoch=4, beam=3
python modified_eval_trans.py --data_folder dataset --dataset_name 'CLEVR' --beam_size 3 --checkpoint result/AdvanceMCCFormers-D/CLEVR/checkpoint_epoch_4first_base_reverse_encode_twice_encode_twice_use_3_imgs.pth.tar --model_name 'AdvanceMCCFormers-D' --info 'base_reverse_CLEVR_epoch4_encode_twice' >eval_results/infos/AdvanceMCCFormers_D_CLEVR_reverse_beamsize_3_encode_twice_epoch4.txt
#epoch=9, beam=3
python modified_eval_trans.py --data_folder dataset --dataset_name 'CLEVR' --beam_size 3 --checkpoint result/AdvanceMCCFormers-D/CLEVR/checkpoint_epoch_9first_base_reverse_encode_twice_encode_twice_use_3_imgs.pth.tar --model_name 'AdvanceMCCFormers-D' --info 'base_reverse_CLEVR_epoch9_encode_twice' >eval_results/infos/AdvanceMCCFormers_D_CLEVR_reverse_beamsize_3_encode_twice_epoch9.txt
# decoder_twice
#epoch=4, beam=3
python modified_eval_trans.py --data_folder dataset --dataset_name 'CLEVR' --beam_size 3 --checkpoint result/AdvanceMCCFormers-D/CLEVR/checkpoint_epoch_4first_base_reverse_encode_twice_encode_twice_use_3_imgs.pth.tar --model_name 'AdvanceMCCFormers-D' --info 'base_reverse_CLEVR_decode_twice_epoch4_encode_twice' --decode_twice 1 >eval_results/infos/AdvanceMCCFormers_D_CLEVR_reverse_beamsize_3_encode_twice_decode_twice_epoch4.txt
#epoch=9, beam=3
python modified_eval_trans.py --data_folder dataset --dataset_name 'CLEVR' --beam_size 3 --checkpoint result/AdvanceMCCFormers-D/CLEVR/checkpoint_epoch_9first_base_reverse_encode_twice_encode_twice_use_3_imgs.pth.tar --model_name 'AdvanceMCCFormers-D' --info 'base_reverse_CLEVR_decode_twice_epoch9_encode_twice' --decode_twice 1 >eval_results/infos/AdvanceMCCFormers_D_CLEVR_reverse_beamsize_3_encode_twice_decode_twice_epoch9.txt

# advance, train CLEVRhat, test CLEVR
#only no reverse
#epoch 4
python modified_eval_trans.py --data_folder dataset --dataset_name 'CLEVR' --beam_size 3 --checkpoint result/AdvanceMCCFormers-D/CLEVRhat/checkpoint_epoch_4first_advance_CLEVRhat_use_2_imgs.pth.tar --model_name 'AdvanceMCCFormers-D' --info 'advance_CLEVRhat_epoch4' > eval_results/infos/AdvanceMCCFormers_D_CLEVR_beamsize_3_epoch4.txt
#epoch 9
python modified_eval_trans.py --data_folder dataset --dataset_name 'CLEVR' --beam_size 3 --checkpoint result/AdvanceMCCFormers-D/CLEVRhat/checkpoint_epoch_9first_advance_CLEVRhat_use_2_imgs.pth.tar --model_name 'AdvanceMCCFormers-D' --info 'advance_CLEVRhat_epoch9' > eval_results/infos/AdvanceMCCFormers_D_CLEVR_beamsize_3_epoch9.txt
#decode twice
python modified_eval_trans.py --data_folder dataset --dataset_name 'CLEVR' --beam_size 3 --checkpoint result/AdvanceMCCFormers-D/CLEVRhat/checkpoint_epoch_4first_advance_CLEVRhat_use_2_imgs.pth.tar --model_name 'AdvanceMCCFormers-D' --info 'advance_CLEVRhat_decode_twice_epoch4' --decode_twice 1 > eval_results/infos/AdvanceMCCFormers_D_CLEVR_beamsize_3_decode_twice_epoch4.txt
#epoch 9
python modified_eval_trans.py --data_folder dataset --dataset_name 'CLEVR' --beam_size 3 --checkpoint result/AdvanceMCCFormers-D/CLEVRhat/checkpoint_epoch_9first_advance_CLEVRhat_use_2_imgs.pth.tar --model_name 'AdvanceMCCFormers-D' --info 'advance_CLEVRhat_decode_twice_epoch9' --decode_twice 1 > eval_results/infos/AdvanceMCCFormers_D_CLEVR_beamsize_3_decode_twice_epoch9.txt
