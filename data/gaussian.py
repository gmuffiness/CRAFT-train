import numpy as np
import cv2

from data.boxEnlarge import enlargebox


class GaussianBuilder(object):
    def __init__(self, init_size, sigma, enlarge_size):
        self.init_size = init_size
        self.sigma = sigma
        self.enlarge_size = enlarge_size
        self.gaussian_map, self.gaussian_map_color = self.generate_gaussian_map()

    def generate_gaussian_map(self):
        circle_mask = self.generate_circle_mask()

        gaussian_map = np.zeros((self.init_size, self.init_size), np.float32)

        for i in range(self.init_size):
            for j in range(self.init_size):
                gaussian_map[i, j] = 1 / 2 / np.pi / (self.sigma ** 2)\
                                     * np.exp(-1 / 2 * ((i - self.init_size / 2) ** 2 / (self.sigma ** 2)
                                                        + (j - self.init_size / 2) ** 2 / (self.sigma ** 2)))




        gaussian_map = gaussian_map * circle_mask
        gaussian_map = (gaussian_map / np.max(gaussian_map)).astype(np.float32)

        gaussian_map_color = (gaussian_map * 255).astype(np.uint8)
        gaussian_map_color = cv2.applyColorMap(gaussian_map_color, cv2.COLORMAP_JET)

        return gaussian_map, gaussian_map_color

    def generate_circle_mask(self):

        zero_arr = np.zeros((self.init_size, self.init_size), np.float32)
        circle_mask = cv2.circle(
            img=zero_arr,
            center=(self.init_size // 2, self.init_size // 2),
            radius=self.init_size // 2,
            color=1,
            thickness=-1,
        )

        return circle_mask

    def four_point_transform(self, bbox):
        """
        Using the pts and the self.gaussian_map, returns Transformed 2d Gaussian map
        """
        width, height = (
            np.max(bbox[:, 0]).astype(np.int32),
            np.max(bbox[:, 1]).astype(np.int32),
        )
        init_points = np.array(
            [
                [0, 0],
                [self.init_size, 0],
                [self.init_size, self.init_size],
                [0, self.init_size],
            ],
            dtype="float32",
        )

        M = cv2.getPerspectiveTransform(init_points, bbox)
        warped_gaussian_map = cv2.warpPerspective(self.gaussian_map, M, (width, height))
        warped_gaussian_map = np.array(warped_gaussian_map * 255, np.uint8)
        return warped_gaussian_map, width, height

    def add_gaussian_map_to_image(self, image, bbox, map_type=None):

        """
        mapping Gaussian heat maps to the character box coordinates of the image.
        """

        # if map_type == "region":
        #     # TODO : edit enlargebox output type from int 32 to float32
        #     bbox = enlargebox(bbox, image.shape[0], image.shape[1], self.enlarge_size)

        bbox = enlargebox(bbox, image.shape[0], image.shape[1], self.enlarge_size)

        if (
            np.any(bbox < 0)
            or np.any(bbox[:, 0] > image.shape[1])
            or np.any(bbox[:, 1] > image.shape[0])
        ):
            return image

        bbox_left, bbox_top = np.array([np.min(bbox[:, 0]), np.min(bbox[:, 1])]).astype(np.int32)
        bbox -= (bbox_left, bbox_top)
        warped_gaussian_map, width, height = self.four_point_transform(
            bbox.astype(np.float32)
        )

        try:
            bbox_area_of_image = image[
                bbox_top : bbox_top + height, bbox_left : bbox_left + width,
            ]
            score_map = np.where(
                warped_gaussian_map > bbox_area_of_image,
                warped_gaussian_map,
                bbox_area_of_image,
            )
            image[
                bbox_top : bbox_top + height, bbox_left : bbox_left + width,
            ] = score_map

        except Exception as e:
            print("Error : {}".format(e))
            print(
                "On generating {} map, strange box came out. (width: {}, height: {})".format(
                    map_type, width, height
                )
            )
        return image

    def calculate_affinity_box_points(self, bbox_1, bbox_2):
        center_1, center_2 = np.mean(bbox_1, axis=0), np.mean(bbox_2, axis=0)
        tl = (bbox_1[0:2].sum(0) + center_1) / 3
        bl = (bbox_2[0:2].sum(0) + center_2) / 3
        tr = (bbox_2[2:4].sum(0) + center_2) / 3
        br = (bbox_1[2:4].sum(0) + center_1) / 3

        # for visualize
        # tl = np.mean([bbox_1[0], bbox_1[1], center_1], axis=0)
        # bl = np.mean([bbox_1[2], bbox_1[3], center_1], axis=0)
        # tr = np.mean([bbox_2[0], bbox_2[1], center_2], axis=0)
        # br = np.mean([bbox_2[2], bbox_2[3], center_2], axis=0)
        bbox = np.array([tl, bl, tr, br]).astype(np.float32)

        return bbox

    def generate_region(self, img_h, img_w, word_level_char_bbox):

        region_map = np.zeros([img_h, img_w], dtype=np.float32)
        for i in range(len(word_level_char_bbox)):
            for j in range(len(word_level_char_bbox[i])):
                region_map = self.add_gaussian_map_to_image(
                    region_map, word_level_char_bbox[i][j].copy(), map_type="region"
                )
        return region_map

    def generate_affinity(self, img_h, img_w, word_level_char_bbox):

        affinity_map = np.zeros([img_h, img_w], dtype=np.float32)
        all_affinity_bbox = []
        for i in range(len(word_level_char_bbox)):
            for j in range(len(word_level_char_bbox[i]) - 1):
                affinity_bbox = self.calculate_affinity_box_points(
                    word_level_char_bbox[i][j], word_level_char_bbox[i][j + 1],
                )

                affinity_map = self.add_gaussian_map_to_image(
                    affinity_map, affinity_bbox.copy(), map_type="affinity"
                )
                all_affinity_bbox.append(np.expand_dims(affinity_bbox, axis=0))

        if len(all_affinity_bbox) > 0:
            all_affinity_bbox = np.concatenate(all_affinity_bbox, axis=0)
        return affinity_map, all_affinity_bbox
