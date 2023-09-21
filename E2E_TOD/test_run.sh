#DST
CUDA_VISIBLE_DEVICES=0,1,2,3 python learn_softmoe.py \
    --data_path_prefix ../data/multiwoz22 \
    --model_name t5-small \
    --pretrained_path ../checkpoints/small \
    --batch_size 8 \
    --gradient_accumulation_steps 2 \
    --dev_batch_size 32 \
    --ckpt_save_path ./ckpt/dst22_small_moe \
    --add_special_decoder_token True \
    --num_experts 16 \
    --slots_per_expert 2 \
    --dst --lr 1e-4 \
    --epoch_num 15 \
    --gpu_ids 0 1 2 3 

'''
# NLG
CUDA_VISIBLE_DEVICES=0,1,2,3 python learn_softmoe.py \
    --data_path_prefix ../data/multiwoz22 \
    --model_name t5-small \
    --pretrained_path ../checkpoints/small \
    --weight_path ./ckpt/dst22_small_moe \
    --batch_size 8 \
    --gradient_accumulation_steps 2 \
    --dev_batch_size 16 \
    --ckpt_save_path ./ckpt/nlg22_small_moe \
    --add_special_decoder_token True \
    --num_experts 32 \
    --slots_per_expert 2 \
    --nlg --lr 1e-4 \
    --epoch_num 15 \
    --warmup_steps 100 \
    --gpu_ids 0 1 2 3
'''