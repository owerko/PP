
import numpy as np
import os
import re
import shutil
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import random
cwd = os.getcwd()
import cv2


crack_images = os.listdir('Positive/')
print("Number of Crack Images: ", len(crack_images))

no_crack_images = os.listdir('Negative/')
print("Number of No Crack Images: ", len(no_crack_images))

## Visualize Random images with cracks
random_indices = np.random.randint(0, len(crack_images), size=4)
print("*****************Random Images with Cracks**************************")
random_images = np.array(crack_images)[random_indices.astype(int)]

f, axarr = plt.subplots(2, 2)
axarr[0, 0].imshow(mpimg.imread(os.path.join(cwd, 'Positive', random_images[0])))
axarr[0, 1].imshow(mpimg.imread(os.path.join(cwd, 'Positive', random_images[1])))
axarr[1, 0].imshow(mpimg.imread(os.path.join(cwd, 'Positive', random_images[2])))
axarr[1, 1].imshow(mpimg.imread(os.path.join(cwd, 'Positive', random_images[3])))
plt.show()

random_indices = np.random.randint(0, len(no_crack_images), size=4)
print("*****************Random Images without Cracks**************************")
random_images = np.array(no_crack_images)[random_indices.astype(int)]

f, axarr = plt.subplots(2, 2)
axarr[0, 0].imshow(mpimg.imread(os.path.join(cwd, 'Negative', random_images[0])))
axarr[0, 1].imshow(mpimg.imread(os.path.join(cwd, 'Negative', random_images[1])))
axarr[1, 0].imshow(mpimg.imread(os.path.join(cwd, 'Negative', random_images[2])))
axarr[1, 1].imshow(mpimg.imread(os.path.join(cwd, 'Negative', random_images[3])))
plt.show()

base_dir = cwd
files = os.listdir(base_dir)


def create_training_data(folder_name):
    train_dir = f"{base_dir}/train/{folder_name}"
    for f in files:
        search_object = re.search(folder_name, f)
        if search_object:
            shutil.move(f'{base_dir}/{folder_name}', train_dir)


create_training_data('Positive')
create_training_data('Negative')

os.makedirs('val/Positive')
os.makedirs('val/Negative')

positive_train = base_dir + "/train/Positive/"
positive_val = base_dir + "/val/Positive/"
negative_train = base_dir + "/train/Negative/"
negative_val = base_dir + "/val/Negative/"

positive_files = os.listdir(positive_train)
negative_files = os.listdir(negative_train)

print(len(positive_files), len(negative_files))

for f in positive_files:
    if random.random() > 0.80:
        shutil.move(f'{positive_train}/{f}', positive_val)


for f in negative_files:
    if random.random() > 0.80:
        shutil.move(f'{negative_train}/{f}', negative_val)