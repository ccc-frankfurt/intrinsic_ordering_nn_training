import os
from urllib.request import urlopen
from io import BytesIO
from zipfile import ZipFile
import subprocess
import glob
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from lib.helpers.dct import *

def visualize_segments(img_path, img, sigma, k, min):
    print('Visualizing segments')

    if len(np.array(img).shape) < 3:
        img_rgb = img.convert('RGB')
    else:
        img_rgb = img

    # https://svaderia.github.io/articles/downloading-and-unzipping-a-zipfile/
    # load the zip with the segmentation algorithm in C
    if not os.path.exists('./metrics/segment'):
        zipurl = 'http://cs.brown.edu/people/pfelzens/segment/segment.zip'
        with urlopen(zipurl) as zipresp:
            with ZipFile(BytesIO(zipresp.read())) as zfile:
                zfile.extractall('metrics/')

        # make the file
        os.system("make -C ./metrics/segment")

    # convert images to pnm
    # and call segment-library with standard parameters (sigma 0.5 k 500 min 20)
    img_rgb.save(img_path + '.ppm')
    subprocess.run(['./metrics/segment/segment', str(sigma), str(k), str(min),
                    img_path + '.ppm', img_path + '_seg' + '_' + str(sigma) + '_' + str(k) + '.ppm'],
                   stdout=subprocess.PIPE)
    res = Image.open(img_path + '_seg' + '_' + str(sigma) + '_' + str(k) + '.ppm')
    res.save(img_path + '_seg' + '_' + str(sigma) + '_' + str(k) +'.png')

def visualize_entropy(img_path, img, window_size):
    print("Visualizing entropy")
    img_arr = np.asarray(img)

    if len(img_arr.shape) > 2:
        gray = 0.2989 * img_arr[:, :, 0] + \
                               0.5870 * img_arr[:, :, 1] + 0.1140 * img_arr[:, :, 2]
    else:
        gray = img_arr
    '''plt.imshow(gray, cmap='gray')
    plt.show()'''
    if gray.shape[0] > window_size and gray.shape[1] > window_size:
        window_entropies = []
        for h in range(gray.shape[0] + 1 - window_size):
            for w in range(gray.shape[1] + 1 - window_size):
                # get window-elements for every img of the batch
                window = gray[h:h + window_size, w:w + window_size]
                values, counts = np.unique(window, return_counts=True)
                probs = counts / float((np.sum(counts)))
                window_entropies.append(-np.sum(probs * np.log2(probs)))

        entropy = np.array(window_entropies).reshape((gray.shape[0] + 1 - window_size,
                                                      gray.shape[1] + 1 - window_size))
        fig, ax = plt.subplots(figsize=(15, 8))
        ax.imshow(entropy, cmap=plt.cm.jet)
        ax.axis('off')
        plt.savefig(img_path + '_entropy_' + str(window_size) + '.png',
                    bbox_inches='tight')

def visualize_img_frequency(img_path, img):
    print("Visualizing frequency")
    img_arr = np.asarray(img)

    if len(img_arr.shape) > 2:
        gray = 0.2989 * img_arr[:, :, 0] + \
               0.5870 * img_arr[:, :, 1] + 0.1140 * img_arr[:, :, 2]
    else:
        gray = img_arr

    coeffs = dct_2d(torch.Tensor(gray/255.), norm='ortho').numpy()
    scale = 0.001
    fig, ax = plt.subplots(figsize=(15, 8))
    im=ax.imshow(coeffs, cmap=plt.cm.jet, vmax=np.max(coeffs)*scale, vmin=np.min(coeffs)*scale)
    cb = fig.colorbar(im)
    min_scaled = np.min(coeffs)*scale
    max_scaled = np.max(coeffs)*scale
    cb.set_ticks(np.linspace(min_scaled, max_scaled, 10))
    min_orig = np.min(coeffs)
    max_orig = np.max(coeffs)
    cb.set_ticklabels([round(i, 2) for i in np.linspace(min_orig, max_orig, 10)])
    ax.axis('off')
    plt.savefig(img_path + '_dct.png',
                bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    # path to images where to compute the metrics
    load_path = './metrics/selected_images'
    save_path = './metrics/selected_images/results'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    files = []
    for ext in ('*.png', '*.jpg', '*.jpeg', '*.JPEG'):
        files.extend(glob.glob(os.path.join(load_path, ext)))

    for path in files:
        print(path)
        img = Image.open(path)
        img_path = os.path.join(save_path, path.split('/')[-1].split('.')[-2])

        for k in [500]: #[200, 300, 400, 500, 600, 700]:
            for sigma in [0.5]: #[0.5, 0.8]:
                visualize_segments(img_path, img, sigma=sigma, k=k, min=20)

        for window_size in [10]: #[5,10,20,30,40,50]:
            visualize_entropy(img_path, img, window_size=window_size)

        visualize_img_frequency(img_path, img)