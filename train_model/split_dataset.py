import os
import shutil
import random
from collections import defaultdict

source_dir = 'all_images'
output_base = 'dataset'
train_ratio = 0.7
val_ratio = 0.15
test_ratio = 0.15

# Get all images and group by class
class_images = defaultdict(list)

for filename in os.listdir(source_dir):
    if filename.endswith('.jpg'):
        label = filename.split('_')[0]
        class_images[label].append(filename)

# Create target directories
for split in ['train', 'val', 'test']:
    for label in class_images:
        split_dir = os.path.join(output_base, split, label)
        os.makedirs(split_dir, exist_ok=True)

# Split and copy
for label, images in class_images.items():
    random.shuffle(images)
    total = len(images)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)

    for i, image in enumerate(images):
        if i < train_end:
            split = 'train'
        elif i < val_end:
            split = 'val'
        else:
            split = 'test'

        src_path = os.path.join(source_dir, image)
        dest_path = os.path.join(output_base, split, label, image)
        shutil.copy(src_path, dest_path)

print("✅ Dataset split into train/val/test under 'dataset/'")
