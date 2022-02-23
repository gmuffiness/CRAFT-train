import os

import numpy as np
import cv2

from config.load_config import cfg
from data.gaussian import GaussianBuilder
from data.boxEnlarge import enlargebox

if __name__ == "__main__":

    vis_path = cfg.vis_test_dir
    TEST_IMG_SAVE_NAME = "peace_score_map.jpg"

    image = np.zeros((464, 1033, 3), dtype=np.uint8)
    img_h, img_w, _ = image.shape

    gaussian_builder = GaussianBuilder(
        cfg.train.data.gaussian.init_size, cfg.train.data.gaussian.sigma
    )

    # word_level_char_bbox = np.array(
    #     [
    #         [[60, 140], [110, 160], [110, 260], [60, 230]],
    #         [[110, 165], [180, 165], [180, 255], [110, 255]],
    #         [[180, 165], [240, 140], [240, 230], [180, 260]],
    #     ]
    # )

    # official paper's PEACE coordinates
    word_level_char_bbox = np.array(
        [
            [[52, 56], [238, 54], [217, 408], [56, 407]],
            [[246, 57], [402, 54], [376, 297], [237, 293]],
            [[414, 37], [577, 38], [542, 299], [381, 297]],
            [[593, 39], [731, 59], [680, 319], [550, 298]],
            [[731, 56], [879, 71], [824, 322], [681, 300]],
        ]
    )

    word_level_char_bbox = word_level_char_bbox[np.newaxis, :, :, :].astype(np.float32)

    region_map = gaussian_builder.generate_region(
        img_h, img_w, word_level_char_bbox
    ).astype(np.uint8)
    region_map_color = cv2.applyColorMap(region_map, cv2.COLORMAP_JET)

    affinity_map, all_affinity_bbox = gaussian_builder.generate_affinity(
        img_h, img_w, word_level_char_bbox
    )
    affinity_map = affinity_map.astype(np.uint8)
    affinity_map_color = cv2.applyColorMap(affinity_map, cv2.COLORMAP_JET)

    region_map_color_vis = region_map_color.copy()
    affinity_map_color_vis = affinity_map_color.copy()
    for word_bbox in word_level_char_bbox:
        for char_bbox in word_bbox:

            # print(f'char_bbox : {char_bbox}')
            # print(char_bbox.dtype, img_h, img_w)
            enlarge = enlargebox(char_bbox.astype(np.int32), img_h, img_w)
            # print(f'enlarged char_bbox : {enlarge}')
            cv2.polylines(
                region_map_color_vis, [np.int32(char_bbox)], True, (0, 0, 255), 2
            )
            cv2.polylines(region_map_color_vis, [enlarge], True, (0, 255, 255), 2)
            cv2.polylines(
                affinity_map_color_vis, [np.int32(char_bbox)], True, (0, 0, 255), 2
            )

    for i in range(len(all_affinity_bbox)):
        cv2.polylines(
            affinity_map_color_vis,
            [all_affinity_bbox[i].astype(np.int)],
            True,
            (255, 0, 255),
            2,
        )

    vis_result = np.vstack(
        [
            np.hstack([region_map_color, affinity_map_color]),
            np.hstack([region_map_color_vis, affinity_map_color_vis]),
        ]
    )

    cv2.line(vis_result, (0, img_h), (img_w * 2, img_h), (255, 255, 255), 2)
    cv2.line(vis_result, (img_w, 0), (img_w, img_h * 2), (255, 255, 255), 2)

    cv2.imwrite(os.path.join(vis_path, TEST_IMG_SAVE_NAME), vis_result)
