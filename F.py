
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image, ImageEnhance
import seaborn as sns
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog
from io import BytesIO
import os


def normalize_brightness_tensor(rawdata, target_mean):
    image = Image.open(BytesIO(rawdata))
    
    # Convert PIL Image to PyTorch Tensor
    transform = transforms.Compose([transforms.ToTensor()])
    tensor_image = transform(image).unsqueeze(0)
    
    
    img_mean = tensor_image.mean().item()

    
    brightness_factor = target_mean / img_mean
    normalized_image = transforms.functional.adjust_brightness(tensor_image, brightness_factor)
    
    
    normalized_image = transforms.ToPILImage()(normalized_image.squeeze(0))
    
    return normalized_image


def display_images(images, titles):
    root = tk.Tk()
    root.title("Normalized Images")

    for i, (image, title) in enumerate(zip(images, titles)):
        tk.Label(root, text=title).grid(row=i, column=0)
        tk.Label(root, image=image).grid(row=i, column=1)

    tk.mainloop()

if __name__ == "__main__":
   
    root = tk.Tk()
    root.withdraw()
    input_dir = filedialog.askdirectory(title="Select Input Directory")
    output_dir = filedialog.askdirectory(title="Select Output Directory")
    target_mean = int(input("Enter the target mean brightness: "))

   
    class CustomDataset(Dataset):
        def __init__(self, input_dir):
            self.files = [os.path.join(input_dir, file) for file in os.listdir(input_dir)]

        def __len__(self):
            return len(self.files)

        def __getitem__(self, idx):
            with open(self.files[idx], 'rb') as file:
                raw_data = file.read()
            return raw_data

    
    dataset = CustomDataset(input_dir)
    dataloader = DataLoader(dataset, batch_size=1)

    normalized_images = []
    original_images = []


    for raw_data in dataloader:
        normalized_image = normalize_brightness_tensor(raw_data[0], target_mean)
        normalized_images.append(normalized_image)
        original_images.append(Image.open(BytesIO(raw_data[0])))

   
    display_images(original_images + normalized_images, ["Original"] * len(original_images) + ["Normalized"] * len(normalized_images))

 
    os.makedirs(output_dir, exist_ok=True)
    for i, normalized_image in enumerate(normalized_images):
        normalized_image.save(os.path.join(output_dir, f"normalized_image_{i}.png"))

    
    original_means = [np.asarray(img).mean() for img in original_images]
    normalized_means = [np.asarray(img).mean() for img in normalized_images]

    sns.set(style="whitegrid")
    plt.figure(figsize=(10, 6))
    sns.barplot(x=["Original"] * len(original_means) + ["Normalized"] * len(normalized_means),
                y=original_means + normalized_means,
                hue=["Image"] * len(original_means) + ["Image"] * len(normalized_means),
                palette=["blue"] * len(original_means) + ["orange"] * len(normalized_means))
    plt.title("Mean Brightness Comparison")
    plt.xlabel("Image Type")
    plt.ylabel("Mean Brightness")
    plt.show()
