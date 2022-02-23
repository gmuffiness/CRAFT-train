import os
import re
import cv2
import argparse
import itertools
import numpy as np
from PIL import Image
from tqdm import tqdm
import multiprocessing
import scipy.io as scio
from copy import deepcopy

from gaussianMap.gaussian import GaussianTransformer


def filter_images(i):
    too_small, too_large, neg_coord, small_affi = 0, 0, 0, 0

    img_dir = os.path.join(synth_dir, name[i][0])
    image = cv2.imread(img_dir, cv2.IMREAD_COLOR)
    max_y, max_x = image.shape[0], image.shape[1]

    img_charbox = deepcopy(charbox[i].transpose(2,1,0))
    for j in range(len(img_charbox)):
        one_charbox = deepcopy(img_charbox[j])
        width = max(one_charbox[:, 0]) - min(one_charbox[:, 0])
        height = max(one_charbox[:, 1]) - min(one_charbox[:, 1])

        # too small width/height of character bboxes
        if width <= 1 or height <= 1:
            too_small += 1
        # negative coordinates of character bboxes
        if np.any(one_charbox < 0):
            neg_coord += 1
        # too large coordinates of character bboxes
        if np.any(one_charbox[:, 0]>max_x) or np.any(one_charbox[:, 1]>max_y):
            too_large += 1

    # too small width / height of affinity bboxes
    total = 0
    for k in range(len(text[i])):
        bboxes = img_charbox[total:total+len(text[i][k])]
        assert len(bboxes) == len(text[i][k])
        total += len(text[i][k])
        for l in range(bboxes.shape[0]-1):
            center_1, center_2 = np.mean(bboxes[l], axis=0), \
                                 np.mean(bboxes[l+1], axis=0)
            tl = (bboxes[l][0:2].sum(0)+center_1) / 3
            bl = (bboxes[l+1][0:2].sum(0)+center_2) / 3
            tr = (bboxes[l+1][2:4].sum(0)+center_2) / 3
            br = (bboxes[l][2:4].sum(0)+center_1) / 3
            affinity = np.array([tl, bl, tr, br]).astype(np.float32)

            width = max(affinity[:, 0]) - min(affinity[:, 0])
            height = max(affinity[:, 1]) - min(affinity[:, 1])

            if width <= 1 or height <= 1:
                small_affi += 1

    if too_small > 0 or neg_coord > 0 or too_large > 0 or small_affi > 0:
        eliminated_idx.append(i)


def generate_gt(i):
    img_dir = os.path.join(synth_dir, name[i][0])
    image = cv2.imread(img_dir, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    img_charbox = deepcopy(charbox[i].transpose(2, 1, 0))

    total = 0
    charbboxes = []
    for j in range(len(text[i])):
        bboxes = img_charbox[total:total+len(text[i][j])]
        assert len(bboxes) == len(text[i][j])
        total += len(text[i][j])
        charbboxes.append(np.array(bboxes))

    enlarge_reg = 0.75
    reg_gauss = GaussianTransformer(imgSize=200, enlargeSize=enlarge_reg)
    region_scores = reg_gauss.generate_region(image.shape,
                                              charbboxes,
                                              signal=img_dir)

    located_folder = name[i][0].split('/')[0]
    path = f"{save_dir}/region/enlarge-{1+enlarge_reg}/"
    if not os.path.exists(os.path.join(path, located_folder)):
        os.makedirs(os.path.join(path, located_folder), exist_ok=True)
    cv2.imwrite(f"{path}/{name[i][0].split('.')[0]}-region.jpg",
                region_scores)

    for enlarge_affi in [0.5, 0.75]:
        affi_gauss = GaussianTransformer(imgSize=200, enlargeSize=enlarge_affi)
        affinity_scores, _ = affi_gauss.generate_affinity(image.shape,
                                                          charbboxes,
                                                          text[i], name[index][0])

        path = f"{save_dir}/affinity/enlarge-{1 + enlarge_affi}/"
        if not os.path.exists(os.path.join(path, located_folder)):
            os.makedirs(os.path.join(path, located_folder), exist_ok=True)
        cv2.imwrite(f"{path}/{name[i][0].split('.')[0]}-affinity.jpg",
                    affinity_scores)




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SynthText Preprocess')
    parser.add_argument('--synth-dir', default='/data/SynthText/',
                        type=str,
                        help='SynthText directory')
    parser.add_argument('--save-dir', default='/data/SynthText-GT/',
                        type=str,
                        help='a directory where GT image will be located')
    args = parser.parse_args()


    # load data
    synth_dir = args.synth_dir
    save_dir = args.save_dir

    # load gt file
    print('============ load gt... ============')
    gt = scio.loadmat(os.path.join(synth_dir, 'gt.mat'))
    charbox = gt['charBB'][0]
    name = gt['imnames'][0]
    text = gt['txt'][0]
    print('=============== done ===============\n')

    # cleaning text
    print('====== start text cleaning... ======')
    for i, txt in enumerate(text):
        txt = [re.split(' \n|\n |\n| ', t.strip()) for t in txt]
        txt = list(itertools.chain(*txt))
        txt = [t for t in txt if len(t) > 0]
        text[i] = txt
    print('=============== done ===============\n')

    eliminated_idx = []
    print('====== start image filtering... ======')
    manager = multiprocessing.Manager()
    eliminated_idx = manager.list()
    p = multiprocessing.Pool(128)
    p.map(filter_images, range(len(charbox)))
    p.close()
    p.join()
    print('=============== done ===============\n')

    # eliminate the filtered image
    print('=== eliminate the filtered image... ===')
    charbox = np.delete(charbox, eliminated_idx)
    name = np.delete(name, eliminated_idx)
    text = np.delete(text, eliminated_idx)
    print('=============== done ===============\n')

    print('======= start generating gt... =======')
    p = multiprocessing.Pool(128)
    p.map(generate_gt, range(len(name)))
    p.close()
    p.join()
    print('=============== done ===============')


