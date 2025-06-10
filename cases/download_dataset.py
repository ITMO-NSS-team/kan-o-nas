from datasets import load_dataset
import os
from PIL import Image  # Ensure Pillow is installed

def download_dataset(path, target_folder):
    # Load the dataset with the correct 'path' argument
    dataset = load_dataset(path)

    # Iterate through each split (e.g., 'train', 'test')
    for split_name in dataset.keys():
        split_dataset = dataset[split_name]
        split_folder = os.path.join(target_folder, split_name)

        # Create split directory if it doesn't exist
        os.makedirs(split_folder, exist_ok=True)

        # Iterate through each example in the split
        for idx, example in enumerate(split_dataset):
            image = example['image']  # PIL Image
            label = example['label']

            # Create class directory (e.g., 'train/5')
            class_folder = os.path.join(split_folder, str(label))
            os.makedirs(class_folder, exist_ok=True)

            # Save image to the class directory
            image_path = os.path.join(class_folder, f"{idx}.png")
            image.save(image_path)

    print(f"Dataset saved to {target_folder} in ImageFolder format.")

# Example usage
download_dataset("ylecun/mnist", "./mnist")
