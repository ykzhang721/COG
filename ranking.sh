
# Training

# baseline: use only entity name
python3 main.py \
    --dataset data_bundle_new_lp.pkl \
    --ptm clip \
    --gpu 1 \
    --save_path ./ckps/ \
    --batch_size 16 \
    --do_train \
    --method onlyent \
    --task ranking


# COG with Concept Integration, using all concepts by default
# python3 main.py \
#     --dataset data_bundle_new_lp.pkl \
#     --ptm clip \
#     --gpu 1 \
#     --con_type all  \
#     --con_loss \
#     --save_path ./ckps/ \
#     --batch_size 16 \
#     --do_train \
#     --method CI \
#     --task ranking


# Evaluation

# baseline: use only entity name
# python3 main.py \
#     --dataset data_bundle_new_lp.pkl \
#     --ptm clip \
#     --gpu 1 \
#     --load_path "" \
#     --batch_size 16 \
#     --do_eval \
#     --method onlyent \
#     --task ranking \


# COG with Concept Integration, using all concepts by default
# python3 main.py \
#     --dataset data_bundle_new_lp.pkl \
#     --ptm clip \
#     --gpu 1 \
#     --load_path "" \
#     --batch_size 16 \
#     --do_eval \
#     --con_type all \
#     --method CI \
#     --task ranking \