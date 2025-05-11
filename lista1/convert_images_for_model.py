import os
import numpy as np
from PIL import Image
import glob

images = []
labels = []

# path to folder with images
folder_path = "/home/kinga/WSI/lista1/numerki/"

for file_path in glob.glob(os.path.join(folder_path, "*.png")):
    image = Image.open(file_path).convert("L")
    image = image.resize((28, 28))
    image_data = np.array(image)
    
    file_name = os.path.basename(file_path)
    label = int(file_name[:1])
    
    images.append(image_data)
    labels.append(label)

images = np.array(images)
labels = np.array(labels)

# saving to file
np.save('images.npy', images)
np.save('labels.npy', labels)

print("Data saved to files 'images.npy' and 'labels.npy'.")
