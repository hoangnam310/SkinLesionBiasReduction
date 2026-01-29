# # Install dependencies
# pip install -r requirements.txt

# Train the cGAN
# python src/train.py \
#     --csv_path dataset/fitzpatrick17k_cleaned.csv \
#     --image_dir dataset/images \
#     --resume outputs/20260129_120733/checkpoints/checkpoint_epoch_0030.pt \
#     --epochs 200

# # Generate synthetic images (after training)
python src/generate.py \
    --checkpoint outputs/20260129_120733/checkpoints/checkpoint_epoch_0030.pt \
    --target_skin_tones 5 6\
    --num_samples 10 \