# Training

# baseline: use only entity name
python3 main.py \
    --dataset data_bundle_new.pkl \
    --ptm clip \
    --batch_size 128 \
    --gpu 0 \
    --save_path ./ckps/ \
    --con_loss \
    --do_train \
    --method onlyent \
    --threshold 0.5 \
    --task classification \
    --epoch 10 

# COG with Concept Integration, using all concepts by default
# python3 main.py \
#     --dataset data_bundle_new.pkl \
#     --ptm clip \
#     --gpu 0 \
#     --save_path ./ckps/ \
#     --con_loss \
#     --do_train \
#     --method CI \
#     --con_type all \
#     --threshold 0.5 \
#     --task classification \
#     --epoch 10


# COG with Concept Integration and Evidence Fusion, using all concepts by default
# python3 main.py \
#     --dataset data_bundle_new.pkl \
#     --ptm clip \
#     --gpu 0 \
#     --save_path ./ckps/ \
#     --con_loss \
#     --do_train \
#     --method CI_EF \
#     --con_type all \
#     --threshold 0.5 \
#     --task classification \
#     --epoch 10


# Ablation: COG with Concept Integration, using blc concepts
# python3 main.py \
#     --dataset data_bundle_new.pkl \
#     --ptm clip \
#     --gpu 0 \
#     --save_path ./ckps/ \
#     --con_loss \
#     --do_train \
#     --method CI \
#     --con_type blc \
#     --threshold 0.5 \
#     --task classification \
#     --epoch 10




# Evaluation

# baseline: use only entity name
# python3 main.py \
#     --dataset data_bundle_new.pkl \
#     --ptm clip \
#     --gpu 0 \
#     --load_path "" \
#     --do_eval \
#     --method onlyent \
#     --threshold 0.5 \
#     --task classification 


# COG with Concept Integration, using all concepts by default
# python3 main.py \
#     --dataset data_bundle_new.pkl \
#     --ptm clip \
#     --gpu 0 \
#     --load_path "" \
#     --do_eval \
#     --method CI \
#     --con_type all \
#     --threshold 0.5 \
#     --task classification 



# COG with Concept Integration and Evidence Fusion, using all concepts by default
# python3 main.py \
#     --dataset data_bundle_new.pkl \
#     --ptm clip \
#     --gpu 1 \
#     --load_path "" \
#     --do_eval \
#     --method CI_EF \
#     --con_type all \
#     --threshold 0.5 \
#     --task classification \
#     --save_evidence


# Ablation: COG with Concept Integration, using blc concepts
# python3 main.py \
#     --dataset data_bundle_new.pkl \
#     --ptm clip \
#     --gpu 0 \
#     --load_path "" \
#     --do_eval \
#     --method CI \
#     --con_type blc \
#     --threshold 0.5 \
#     --task classification 
