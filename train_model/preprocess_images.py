import pandas as pd
import os
import shutil

# Load metadata CSV
metadata = pd.read_csv('HAM10000_metadata.csv')

# Combine image folders
image_folder1 = 'images_part_1'
image_folder2 = 'images_part_2'
all_images_dir = 'all_images'

# Create target folder
os.makedirs(all_images_dir, exist_ok=True)

# Combine image paths
image_paths = {}
for folder in [image_folder1, image_folder2]:
    for fname in os.listdir(folder):
        image_paths[fname] = os.path.join(folder, fname)

# Map disease codes to readable names
label_map = {
    'nv': 'Melanocytic_nevi',
    'mel': 'Melanoma',
    'bkl': 'Benign_keratosis',
    'bcc': 'Basal_cell_carcinoma',
    'akiec': 'Actinic_keratoses',
    'vasc': 'Vascular_lesions',
    'df': 'Dermatofibroma'
}

# Filter and preprocess images
count = 0
for index, row in metadata.iterrows():
    image_id = row['image_id']
    label_code = row['dx']

    if label_code in label_map:
        filename = image_id + '.jpg'
        if filename in image_paths:
            label = label_map[label_code]
            target_path = os.path.join(all_images_dir, f"{label}_{count}.jpg")
            shutil.copy(image_paths[filename], target_path)
            count += 1

print(f"✅ Copied and labeled {count} images to 'all_images/'")
