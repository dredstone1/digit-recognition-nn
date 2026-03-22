from torchvision import datasets, transforms
import torch
import matplotlib.pyplot as plt

transform = transforms.Compose([
    # transforms.Lambda(lambda img: transforms.functional.rotate(img, 90)),  # fix EMNIST rotation
    # transforms.Lambda(lambda img: transforms.functional.hflip(img)),
    transforms.ToTensor()
])
train = datasets.EMNIST(
    root="data",
    split="balanced",
    train=True,
    download=True,
    transform=transform
)

test = datasets.EMNIST(
    root="data",
    split="balanced",
    train=False,
    download=True,
    transform=transform
)

# ---- WRITE TRAIN FILE ----
with open("../ModelData/emnist_letters_train.nndb", "w") as f:
    f.write(f"{len(train)} 784\n")
    
    for idx, (image, label) in enumerate(train):
        flat = image.reshape(-1).tolist()
        line = f"p{label} " + " ".join(map(str, flat)) + "\n"
        f.write(line)

# ---- WRITE TEST FILE ----
with open("../ModelData/emnist_letters_test.nndb", "w") as f:
    f.write(f"{len(test)} 784\n")
    
    for idx, (image, label) in enumerate(test):
        flat = image.reshape(-1).tolist()
        line = f"p{label} " + " ".join(map(str, flat)) + "\n"
        f.write(line)

# ---- SHOW SAMPLE ----
index_to_show = 0
print("fff")

# while True:
#     sample_image, sample_label = train[index_to_show]
#
#     # Fix EMNIST rotation
#     sample_image = torch.rot90(sample_image, 2, [1, 2])
#     # if (sample_label != 1):
#     #     index_to_show += 0
#     #     continue
#
#     print("Sample index:", index_to_show)
#     print("Sample label:", sample_label)
#     print("Sample image shape:", sample_image.shape)
#     plt.imshow(sample_image.squeeze(), cmap="gray")
#     plt.title(f"Index: {index_to_show}  |  Label: {sample_label}")
#     plt.show()
#     index_to_show += 1
#
