import os
import shutil
from sklearn.model_selection import train_test_split

# Current raw output folders
IMAGE_DIR = 'data/processed/images'
LABEL_DIR = 'data/processed/labels'

# New YOLO-compliant folders
YOLO_DIR = 'data/yolo_dataset'

def setup_yolo_folders():
    print("Creating YOLO folder structure...")
    for split in ['train', 'val']:
        os.makedirs(os.path.join(YOLO_DIR, 'images', split), exist_ok=True)
        os.makedirs(os.path.join(YOLO_DIR, 'labels', split), exist_ok=True)
        
    # Grab all the images we just generated
    images = [f for f in os.listdir(IMAGE_DIR) if f.endswith('.jpg')]
    
    print(f"Found {len(images)} total images. Splitting 80/20...")
    
    # Classic 80/20 split using scikit-learn
    train_imgs, val_imgs = train_test_split(images, test_size=0.2, random_state=42)
    
    def copy_files(file_list, split_name):
        print(f"Copying {len(file_list)} files to {split_name} set...")
        for img_name in file_list:
            # Copy Image
            shutil.copy(
                os.path.join(IMAGE_DIR, img_name), 
                os.path.join(YOLO_DIR, 'images', split_name, img_name)
            )
            # Copy matching Label
            label_name = img_name.replace('.jpg', '.txt')
            if os.path.exists(os.path.join(LABEL_DIR, label_name)):
                shutil.copy(
                    os.path.join(LABEL_DIR, label_name), 
                    os.path.join(YOLO_DIR, 'labels', split_name, label_name)
                )

    copy_files(train_imgs, 'train')
    copy_files(val_imgs, 'val')
    print("Done! Dataset is formatted and ready for YOLOv8.")

if __name__ == "__main__":
    setup_yolo_folders()