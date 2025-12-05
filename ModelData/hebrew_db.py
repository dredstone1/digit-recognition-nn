import os
from PIL import Image, ImageOps
from datasets import load_dataset
from tqdm import tqdm
import random

def process_and_remap_dataset(dataset_split, output_filename, duplicate_times=1):
    TARGET_WIDTH, TARGET_HEIGHT = 28, 28
    LABEL_OFFSET = 46
    processed_lines = []

    for sample in tqdm(dataset_split, desc=f"Processing images for {output_filename}"):
        original_label = sample['label']

        new_label = original_label + LABEL_OFFSET
        image = sample['image'].convert('L')
        bbox = image.getbbox()
        cropped = image.crop(bbox) if bbox else Image.new('L', image.size, 0)

        for dup_index in range(duplicate_times):
            shrink_size = random.randint(15, 28)
            resized = cropped.resize((shrink_size, shrink_size), Image.LANCZOS)

            # Center on 28x28 canvas
            canvas = Image.new('L', (TARGET_WIDTH, TARGET_HEIGHT), 255)
            offset = ((TARGET_WIDTH - shrink_size) // 2, (TARGET_HEIGHT - shrink_size) // 2)
            canvas.paste(resized, offset)

            # Rotate 90° clockwise and mirror
            rotated = canvas.rotate(-90, expand=True)
            mirrored = ImageOps.mirror(rotated)
            final_image = ImageOps.invert(mirrored)

            # Convert to normalized pixel string
            pixels = list(final_image.getdata())
            normalized = [p / 255.0 for p in pixels]
            line = f"w1 p{new_label} {' '.join(map(str, normalized))}\n"
            processed_lines.append(line)

    # Write NNDB file
    with open(output_filename, 'w') as f:
        f.write(f"{len(processed_lines)} {TARGET_WIDTH * TARGET_HEIGHT}\n")
        f.writelines(processed_lines)

    print(f"\nSaved {len(processed_lines)} samples to {output_filename}")

if __name__ == "__main__":
    print("Loading hebrew-handwritten-dataset...")
    ds = load_dataset("sivan22/hebrew-handwritten-dataset")
    print("Dataset loaded successfully.\n")

    duplicate_times =10  # Number of duplicates per sample
    if 'train' in ds:
        process_and_remap_dataset(ds['train'], 'hebrew_train.nndb', duplicate_times)
    if 'test' in ds:
        process_and_remap_dataset(ds['test'], 'hebrew_test.nndb', duplicate_times)

    print("\nConversion complete: images remapped, inverted, and duplicated with progressive shrink.")

