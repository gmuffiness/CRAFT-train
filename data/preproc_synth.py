import scipy.io as scio
import os
import torch.utils.data as data
import torchvision.transforms as transforms
import cv2
import numpy as np
import re
import itertools
from PIL import Image

from data import imgproc
from utils import config

from data.gaussian import GaussianTransformer
from data.imgaug import random_scale_for_synth, random_crop






class SynthTextDataLoader(data.Dataset):
    def __init__(self, target_size=768, data_dir_list={"synthtext":"datapath"}, viz=False, mode=''):
        assert 'synthtext' in data_dir_list.keys()

        self.target_size = target_size
        self.data_dir_list = data_dir_list
        self.viz = viz
        self.charbox, self.image, self.imgtxt = self.load_synthtext(mode)
        self.gen = GaussianTransformer(200, 1.5)
        # self.gen.gen_circle_mask()


    def choice_train_test_split(self,X, test_size=0.1, shuffle=True, random_state=1004):

        test_num = int(X.shape[0] * test_size)
        train_num = X.shape[0] - test_num

        if shuffle:
            np.random.seed(random_state)
            train_idx = np.random.choice(X.shape[0], train_num, replace=False)
            # -- way 1: using np.setdiff1d()
            test_idx = np.setdiff1d(range(X.shape[0]), train_idx)

            X_train = X[train_idx]
            X_test = X[test_idx]

        else:
            X_train = X[:train_num]
            X_test = X[train_num:]

        return X_train, X_test


    def load_synthtext(self, mode):

        gt = scio.loadmat(os.path.join(self.data_dir_list["synthtext"], 'gt.mat'))
        wordbox = gt['wordBB'][0]
        charbox = gt['charBB'][0]
        image = gt['imnames'][0]
        imgtxt = gt['txt'][0]

        trn_wordbox, tst_wordbox = self.choice_train_test_split(wordbox)
        trn_charbox, tst_charbox = self.choice_train_test_split(charbox)
        trn_image, tst_image = self.choice_train_test_split(image)
        trn_imgtxt, tst_imgtxt = self.choice_train_test_split(imgtxt)


        if mode == 'train':
            return trn_charbox, trn_image, trn_imgtxt

        elif mode == 'test':
            return tst_wordbox, tst_image, tst_imgtxt
        else:
            return charbox, image, imgtxt


    def load_synthtext_image_gt(self, index):
        img_path = os.path.join(self.data_dir_list["synthtext"], self.image[index][0])
        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        _charbox = self.charbox[index].transpose((2, 1, 0))
        image = random_scale_for_synth(image, _charbox, self.target_size)
        words = [re.split(' \n|\n |\n| ', t.strip()) for t in self.imgtxt[index]]
        words = list(itertools.chain(*words))
        words = [t for t in words if len(t) > 0]

        character_bboxes = []
        total = 0
        confidences = []
        for i in range(len(words)):
            bboxes = _charbox[total:total + len(words[i])]
            assert len(bboxes) == len(words[i])
            total += len(words[i])
            bboxes = np.array(bboxes)
            character_bboxes.append(bboxes)
            confidences.append(1.0)

        return image, character_bboxes, words, np.ones((image.shape[0], image.shape[1]), np.float32), confidences, img_path



