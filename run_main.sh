# # Install dependencies
# pip install -r requirements.txt

# Train the cGAN
# python src/train.py \
#     --csv_path dataset/fitzpatrick17k_cleaned.csv \
#     --image_dir dataset/images \
#     --resume outputs/20260129_131055/checkpoints/checkpoint_epoch_0040.pt \
#     --epochs 200

# # Generate synthetic images (after training)
# python src/generate.py \
#     --checkpoint outputs/20260129_120733/checkpoints/checkpoint_epoch_0030.pt \
#     --target_skin_tones 5 6\
#     --num_samples 10 \

# See TensorBoard
# tensorboard --logdir [outputs/20260129_120733]

# Train WGAN-GP
# python src/train_wgan.py \
#     --epochs 100 \
#     --batch_size 64 \
#     --lr_g 0.0001 \
#     --lr_c 0.0001 \
#     --n_critic 5 \
#     --lambda_gp 10.0 \
#     --beta1 0.0 \
#     --beta2 0.9

# To resume training from a checkpoint:
python src/train_wgan.py \
    --csv_path dataset/fitzpatrick17k_cleaned.csv \
    --image_dir dataset/images \
    --resume outputs/wgan_20260131_230948/checkpoints/final_model.pt \
    --epochs 500\
    --batch_size 64 \
    --lr_g 0.0001 \
    --lr_c 0.0001 \
    --n_critic 5 \
    --lambda_gp 10.0 \
    --beta1 0.0 \
    --beta2 0.9