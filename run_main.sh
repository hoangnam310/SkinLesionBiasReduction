# # Install dependencies
# pip install -r requirements.txt

# Train the cGAN
python src/train.py \
    --csv_path dataset/fitzpatrick17k_cleaned.csv \
    --image_dir dataset/images \
    --epochs 200 \
    --batch_size 64

# # Generate synthetic images (after training)
# python src/generate.py \
#     --checkpoint outputs/<timestamp>/checkpoints/final_model.pt \
#     --target_skin_tones 5 6 \
#     --num_samples 500 \
#     --create_csv