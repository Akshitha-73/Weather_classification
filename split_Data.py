import os
import shutil
from sklearn.model_selection import train_test_split

input_dir = "feature_extraction\data\Multi-class Weather Dataset"
output_dir = "dataset"

splits = ["train", "val"]
split_ratio = 0.2  

for cls in os.listdir(input_dir):
    cls_path = os.path.join(input_dir, cls)
    if not os.path.isdir(cls_path):
        continue
    
    images = [os.path.join(cls_path, img) for img in os.listdir(cls_path)]
    
    train_imgs, val_imgs = train_test_split(images, test_size=split_ratio, random_state=42)

    for split, split_imgs in zip(splits, [train_imgs, val_imgs]):
        split_cls_dir = os.path.join(output_dir, split, cls)
        os.makedirs(split_cls_dir, exist_ok=True)
        for img_path in split_imgs:
            shutil.copy(img_path, split_cls_dir)

print("✅ Dataset split complete! Check the 'dataset' folder.")
