import os
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import numpy as np
from collections import Counter
import random

class CSVGenerator:
    def script_to_generate_csv(self,fileName,base_dir):
        # Define the directory for the augmented_train dataset
        # base_dir = "../data/augmented_train"

        # Create a list to hold the rows of the CSV
        data_rows = []

        # Walk through the directories
        for label in os.listdir(base_dir):
            label_path = os.path.join(base_dir, label)
            if os.path.isdir(label_path):  # Check if it's a directory
                for image_name in os.listdir(label_path):
                    image_path = os.path.join(label, image_name)  # Relative path for the image
                    data_rows.append([image_path, label])

        # Create a DataFrame
        df = pd.DataFrame(data_rows, columns=["image_path", "label"])

        # Shuffle the DataFrame rows
        df = df.sample(frac=1).reset_index(drop=True)

        # Add an ID column
        df.insert(0, "id", range(1, len(df) + 1))

        # Save the DataFrame to a CSV file
        output_csv_path = f"../data/{fileName}.csv"
        df.to_csv(output_csv_path, index=False)

        print(f"CSV file has been saved at: {output_csv_path}")




class CIFAR10PreProcessor:
    def __init__(self, train_labels_path, train_images_dir, test_images_dir):
        """
        Initialize the CIFAR10 data processor

        Args:
            train_labels_path: Path to training labels CSV
            train_images_dir: Directory containing training images
            test_images_dir: Directory containing test images
        """
        self.train_labels_path = train_labels_path
        self.train_images_dir = train_images_dir
        self.test_images_dir = test_images_dir
        self.train_labels = None
        self.load_and_analyze_dataset()


    def load_and_analyze_dataset(self):
        """Load and analyze the dataset, print basic statistics"""
        # Load labels
        self.train_labels = pd.read_csv(self.train_labels_path)

        # Convert id column to string and add .png extension
        self.train_labels['id'] = self.train_labels['id'].astype(str) + '.png'

        # Print basic statistics
        print("Dataset Overview:")
        print(f"Number of training samples: {len(self.train_labels)}")
        print(f"Number of training images: {len(os.listdir(self.train_images_dir))}")
        print(f"Number of testing images: {len(os.listdir(self.test_images_dir))}")

        # Class distribution
        class_dist = Counter(self.train_labels['label'])
        print("\nClass Distribution:")
        for label, count in class_dist.items():
            print(f"{label}: {count} images ({count / len(self.train_labels) * 100:.2f}%)")

    def normalize_image(self, img):
        """Normalize image pixel values to range [0,1]"""
        return img.astype('float32') / 255.0

    def augment_image(self, img, rotation=True, flip=True, brightness=True, zoom=True):
        """Apply various augmentation techniques to an image"""
        augmented = img.copy()

        if rotation:
            angle = np.random.uniform(-15, 15)
            height, width = img.shape[:2]
            center = (width / 2, height / 2)
            rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            augmented = cv2.warpAffine(augmented, rotation_matrix, (width, height))

        if flip and np.random.random() > 0.5:
            augmented = cv2.flip(augmented, 1)

        if brightness:
            beta = np.random.uniform(-30, 30)
            augmented = cv2.convertScaleAbs(augmented, beta=beta)

        if zoom:
            scale = np.random.uniform(0.8, 1.2)
            height, width = img.shape[:2]
            center = (width / 2, height / 2)
            zoom_matrix = cv2.getRotationMatrix2D(center, 0, scale)
            augmented = cv2.warpAffine(augmented, zoom_matrix, (width, height))

        return augmented

    def load_and_display_images(self, num_samples=5, random_state=None, normalize=True):
        """Display sample images from the dataset"""
        if random_state is not None:
            samples = self.train_labels.sample(n=num_samples, random_state=random_state)
        else:
            samples = self.train_labels.head(num_samples)

        fig, axes = plt.subplots(1, num_samples, figsize=(15, 5))

        for i in range(num_samples):
            img_name = samples.iloc[i]['id']
            label = samples.iloc[i]['label']
            img_path = os.path.join(self.train_images_dir, img_name)

            try:
                img = cv2.imread(img_path)
                if img is None:
                    raise ValueError(f"Failed to load image: {img_path}")
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                orig_img = img.copy()
                if normalize:
                    img = self.normalize_image(img)

                brightness = np.mean(cv2.cvtColor(orig_img, cv2.COLOR_RGB2GRAY))

                axes[i].imshow(img)
                axes[i].set_title(f"Label: {label}\nBrightness: {brightness:.1f}")
                axes[i].axis("off")

            except Exception as e:
                print(f"Error loading image {img_name}: {str(e)}")
                axes[i].text(0.5, 0.5, "Error loading image", ha='center')
                axes[i].axis("off")

        plt.tight_layout()
        plt.show()
        return fig

    def analyze_image_properties(self, sample_size=100, normalize=True):
        """Analyze properties of images in the dataset"""
        samples = self.train_labels.sample(n=min(sample_size, len(self.train_labels)))

        sizes = []
        brightness_values = []
        pixel_value_ranges = []
        corrupted = 0

        for _, row in samples.iterrows():
            img_path = os.path.join(self.train_images_dir, row['id'])
            try:
                img = cv2.imread(img_path)
                if img is None:
                    corrupted += 1
                    continue

                orig_img = img.copy()
                if normalize:
                    img = self.normalize_image(img)

                sizes.append(img.shape)
                brightness = np.mean(cv2.cvtColor(orig_img, cv2.COLOR_RGB2GRAY))
                brightness_values.append(brightness)

                pixel_value_ranges.append({
                    'min': img.min(),
                    'max': img.max(),
                    'mean': img.mean()
                })

            except Exception:
                corrupted += 1

        self._print_analysis_results(sample_size, corrupted, sizes, brightness_values,
                                     pixel_value_ranges, normalize)

        return sizes, brightness_values, pixel_value_ranges

    def _print_analysis_results(self, sample_size, corrupted, sizes, brightness_values,
                                pixel_value_ranges, normalize):
        """Helper method to print analysis results"""
        print("\nImage Analysis:")
        print(f"Sample size: {sample_size}")
        print(f"Corrupted images: {corrupted}")
        print(f"Unique image sizes: {set(sizes)}")
        print(f"Average brightness (original): {np.mean(brightness_values):.2f}")

        if normalize:
            pixel_stats = pd.DataFrame(pixel_value_ranges)
            print("\nNormalized Pixel Value Statistics:")
            print(f"Min: {pixel_stats['min'].mean():.3f}")
            print(f"Max: {pixel_stats['max'].mean():.3f}")
            print(f"Mean: {pixel_stats['mean'].mean():.3f}")

    def display_augmentations(self, img_index=0, num_augmentations=5):
        """Display original and augmented versions of an image"""
        img_path = os.path.join(self.train_images_dir, self.train_labels.iloc[img_index]['id'])
        img = cv2.imread(img_path)

        fig, axes = plt.subplots(1, num_augmentations + 1, figsize=(20, 4))

        axes[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        axes[0].set_title('Original')
        axes[0].axis('off')

        for i in range(num_augmentations):
            augmented = self.augment_image(img)
            axes[i + 1].imshow(cv2.cvtColor(augmented, cv2.COLOR_BGR2RGB))
            axes[i + 1].set_title(f'Augmented {i + 1}')
            axes[i + 1].axis('off')

        plt.tight_layout()
        plt.show()

    def create_augmented_dataset(self, subset_size=None, augmentations_per_image=3):
        """Create augmented dataset"""
        if subset_size:
            labels_df = self.train_labels.head(subset_size)
        else:
            labels_df = self.train_labels

        augmented_dataset = []

        for idx, row in labels_df.iterrows():
            img_path = os.path.join(self.train_images_dir, row['id'])
            try:
                img = cv2.imread(img_path)
                if img is None:
                    continue

                augmented_dataset.append((img, row['label']))

                for _ in range(augmentations_per_image):
                    aug_img = self.augment_image(img)
                    augmented_dataset.append((aug_img, row['label']))

            except Exception as e:
                print(f"Error processing image {row['id']}: {str(e)}")
                continue

            if idx % 1000 == 0:
                print(f"Processed {idx} images...")

        print(f"\nTotal dataset size after augmentation: {len(augmented_dataset)}")
        return augmented_dataset

    def create_and_save_augmented_images(self, augmented_dir="../data/augmented_train", augmentations_per_image=4):
        """
        Create and save augmented versions of all images in a new directory

        Args:
            augmented_dir: Directory where augmented images will be saved
            augmentations_per_image: Number of augmented versions to create per original image
        """
        # Create augmented directory if it doesn't exist
        os.makedirs(augmented_dir, exist_ok=True)

        total_images = len(self.train_labels)

        for idx, row in self.train_labels.iterrows():
            # Create subdirectory for each class
            class_dir = os.path.join(augmented_dir, row['label'])
            os.makedirs(class_dir, exist_ok=True)

            img_path = os.path.join(self.train_images_dir, row['id'])
            try:
                # Load original image
                img = cv2.imread(img_path)
                if img is None:
                    continue

                # Save original image
                original_filename = f"original_{row['id']}"
                cv2.imwrite(os.path.join(class_dir, original_filename), img)

                # Create and save augmented versions
                for aug_idx in range(augmentations_per_image):
                    aug_img = self.augment_image(img)
                    aug_filename = f"aug{aug_idx + 1}_{row['id']}"
                    cv2.imwrite(os.path.join(class_dir, aug_filename), aug_img)

            except Exception as e:
                print(f"Error processing image {row['id']}: {str(e)}")
                continue

            # Print progress every 1000 images
            if idx % 1000 == 0:
                progress = (idx / total_images) * 100
                print(f"Processed {idx}/{total_images} images ({progress:.2f}%)...")

        print("\nAugmentation complete!")
        print(f"Images saved in: {augmented_dir}")

        # Print directory structure summary
        class_counts = {}
        for class_name in os.listdir(augmented_dir):
            class_path = os.path.join(augmented_dir, class_name)
            if os.path.isdir(class_path):
                num_images = len(os.listdir(class_path))
                class_counts[class_name] = num_images

        print("\nAugmented Dataset Summary:")
        for class_name, count in class_counts.items():
            print(f"{class_name}: {count} images")

    def prepare_training_dataset(self, output_dir="../data/final_train_dataset"):
        """
        Prepare a combined training dataset from original and augmented images

        Args:
            output_dir: Directory where the final training dataset will be created

        Returns:
            DataFrame containing paths and labels for the new training dataset
        """
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Initialize list to store image information
        dataset_info = []

        # Process original training images
        print("Processing original training images...")
        for idx, row in self.train_labels.iterrows():
            try:
                # Load and process original image
                orig_path = os.path.join(self.train_images_dir, row['id'])
                if os.path.exists(orig_path):
                    # Create class directory if it doesn't exist
                    class_dir = os.path.join(output_dir, row['label'])
                    os.makedirs(class_dir, exist_ok=True)

                    # Copy original image with new name
                    new_filename = f"orig_{row['id']}"
                    new_path = os.path.join(class_dir, new_filename)
                    img = cv2.imread(orig_path)
                    if img is not None:
                        cv2.imwrite(new_path, img)
                        dataset_info.append({
                            'path': new_path,
                            'label': row['label'],
                            'type': 'original'
                        })
            except Exception as e:
                print(f"Error processing original image {row['id']}: {str(e)}")
                continue

            if idx % 1000 == 0:
                print(f"Processed {idx} original images...")

        # Process augmented images
        print("\nProcessing augmented images...")
        augmented_dir = "../data/augmented_train"

        for class_name in os.listdir(augmented_dir):
            class_path = os.path.join(augmented_dir, class_name)
            if not os.path.isdir(class_path):
                continue

            # Create class directory in output
            output_class_dir = os.path.join(output_dir, class_name)
            os.makedirs(output_class_dir, exist_ok=True)

            # Process augmented images for this class
            for img_name in os.listdir(class_path):
                if img_name.startswith('aug'):  # Only process augmented images
                    try:
                        aug_path = os.path.join(class_path, img_name)
                        new_path = os.path.join(output_class_dir, img_name)

                        img = cv2.imread(aug_path)
                        if img is not None:
                            cv2.imwrite(new_path, img)
                            dataset_info.append({
                                'path': new_path,
                                'label': class_name,
                                'type': 'augmented'
                            })
                    except Exception as e:
                        print(f"Error processing augmented image {img_name}: {str(e)}")
                        continue

        # Create DataFrame with dataset information
        dataset_df = pd.DataFrame(dataset_info)

        # Save dataset information to CSV
        csv_path = os.path.join(output_dir, 'training_dataset_info.csv')
        dataset_df.to_csv(csv_path, index=False)

        # Print summary statistics
        print("\nDataset Summary:")
        print(f"Total images: {len(dataset_df)}")
        print("\nImages per class:")
        class_counts = dataset_df['label'].value_counts()
        for label, count in class_counts.items():
            print(f"{label}: {count}")

        print("\nImages by type:")
        type_counts = dataset_df['type'].value_counts()
        for type_name, count in type_counts.items():
            print(f"{type_name}: {count}")

        print(f"\nDataset information saved to: {csv_path}")

        return dataset_df