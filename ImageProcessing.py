import numpy as np
import imagesize
import io, os
import cv2
from matplotlib import pyplot as plt
from PIL import Image
import torchvision

def determine_sizes(img_dir):

    all_img = os.listdir(img_dir)
    for img in all_img:
        if (not "jpg" in img):
            all_img.remove(img)
            
    sizes = np.empty((len(all_img), 2))

    for i in range(len(all_img)):
        sizes[i] = np.array(imagesize.get(img_dir + all_img[i]))

    x_dim = sizes[:,0]
    min_x = min(x_dim)
    max_x = max(x_dim)

    y_dim = sizes[:,1]
    min_y = min(y_dim)
    max_y = max(y_dim)
    
    xy_dim=np.prod(sizes,axis=1)
    min_xy=np.min(xy_dim)
    max_xy=np.max(xy_dim)
    return min_y, max_y, min_x, max_x,min_xy,max_xy, sizes

def load_show_img(path):
    img = cv2.imread(path)
    plt.imshow(img)
    plt.show()

def show_img(img):
    plt.imshow(img)
    plt.show()
    
def show_img_gray(img):
    plt.imshow(img, cmap="gray", vmin=0, vmax=255)
    plt.show()

def load_img_rgb(path):
    return cv2.imread(path, cv2.IMREAD_COLOR)

def load_img_gray(path):
    return cv2.imread(path, cv2.IMREAD_GRAYSCALE)

def load_img_colour(path):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    RGB_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return RGB_img

def resize_images(path, new_path, size):
    gray = False
    for image in os.listdir(path):
        if (not "jpg" in image):
            continue
        im = Image.open(os.path.join(path, image))
        imResize = torchvision.transforms.functional.resize(
            im, size, interpolation=Image.LANCZOS
            )
        if gray:
            imResize = torchvision.transforms.functional.to_grayscale(imResize)
        
        #delete_path = os.path.join(path, image)
        #os.remove(delete_path)
        save_path = os.path.join(new_path, image)
        imResize.save(save_path)