

def make_pseudo_char_box(self, net, image, word_bbox, word, vis_opt=False, img_name=''):
    # print('inference_pursedo_bboxes model last parameters
    # :{}'.format(net.module.conv_cls[-1].weight.reshape(2, -1)))
    if net.training:
        net.eval()
    with torch.no_grad():
        word_image, MM = self.crop_image_by_bbox(image, word_bbox)

        real_word_without_space = word.replace('\s', '')
        real_char_nums = len(real_word_without_space)
        input = word_image.copy()
        # 왜 64로 scale 조절을 하는 걸까?? --> https://github.com/clovaai/CRAFT-pytorch/issues/18
        scale = 64.0 / input.shape[0]
        # input = cv2.resize(input, None, fx=scale, fy=scale)
        input = cv2.resize(input, None, fx=scale, fy=scale)
        input_copy = input.copy()

        img_torch = torch.from_numpy(imgproc.normalizeMeanVariance(input, mean=(0.485, 0.456, 0.406),
                                                                   variance=(0.229, 0.224, 0.225)))
        img_torch = img_torch.permute(2, 0, 1).unsqueeze(0)
        img_torch = img_torch.type(torch.FloatTensor).cuda()
        scores, _ = net(img_torch)
        region_score = scores[0, :, :, 0].cpu().data.numpy()
        region_score = np.uint8(np.clip(region_score, 0, 1) * 255)
        bgr_region_scores = cv2.resize(region_score, (input.shape[1], input.shape[0]))
        bgr_region_scores = cv2.cvtColor(bgr_region_scores, cv2.COLOR_GRAY2RGB)

        pursedo_bboxes, color_markers = watershed_v4(bgr_region_scores.copy(), input.copy(), vis_opt=False)

        if len(pursedo_bboxes) > 0:
            pursedo_bboxes[:, :, 0] = np.clip(pursedo_bboxes[:, :, 0], 0, bgr_region_scores.shape[1])
            pursedo_bboxes[:, :, 1] = np.clip(pursedo_bboxes[:, :, 1], 0, bgr_region_scores.shape[0])

        _tmp = []
        # except for the small box
        for i in range(pursedo_bboxes.shape[0]):
            if np.mean(pursedo_bboxes[i].ravel()) > 2:  # ravel -> 1차원 변환
                _tmp.append(pursedo_bboxes[i])
            else:
                print("filter bboxes", pursedo_bboxes[i])  # 작은 box들

            # check small box 2
            # import ipdb;ipdb.set_trace()
            #
            # poly = plg.Polygon(pursedo_bboxes[i])
            # area = poly.area()
            # if area < 10:
            #     continue
            # _tmp.append(pursedo_bboxes[i])
            #
            #
            #
            #
            #
            # pursedo_bboxes_ = pursedo_bboxes[i].copy()
            # top_left = np.array([np.min(pursedo_bboxes[i][:, 0]), np.min(pursedo_bboxes[i][:, 1])]).astype(np.int32)
            # pursedo_bboxes_ -= top_left[None, :]
            #
            # width, height = np.max(pursedo_bboxes_[:, 0]).astype(np.int32), np.max(
            #     pursedo_bboxes_[:, 1]).astype(np.int32)
            #
            # if width >0 or height >0:
            #     _tmp.append(pursedo_bboxes[i])
            # else:
            #     import ipdb;ipdb.set_trace()
            #     print("filter bboxes", pursedo_bboxes[i])  # 작은 box들

        pursedo_bboxes = np.array(_tmp, np.float32)
        if pursedo_bboxes.shape[0] > 1:
            index = np.argsort(pursedo_bboxes[:, 0, 0])
            pursedo_bboxes = pursedo_bboxes[index]

        confidence = self.get_confidence(real_char_nums, len(pursedo_bboxes))

        bboxes = []
        if confidence <= 0.5:  # confidence 값들이 낮은 경우 등분하고, 이떄 confidence 0.5
            width = input.shape[1]
            height = input.shape[0]

            width_per_char = width / len(word)
            for j, char in enumerate(word):
                if char == ' ':
                    continue
                left = j * width_per_char
                right = (j + 1) * width_per_char
                bbox = np.array([[left, 0], [right, 0], [right, height],
                                 [left, height]])
                bboxes.append(bbox)

            bboxes = np.array(bboxes, np.float32)
            confidence = 0.5

        else:
            bboxes = pursedo_bboxes

        if vis_opt == True:

            # -----------------------------------------------------------------------------------------------#

            input_copy1 = input_copy.copy()
            _purs_bboxes = np.int32(pursedo_bboxes.copy())
            if len(_purs_bboxes) > 0:
                _purs_bboxes[:, :, 0] = np.clip(_purs_bboxes[:, :, 0], 0, input.shape[1])
                _purs_bboxes[:, :, 1] = np.clip(_purs_bboxes[:, :, 1], 0, input.shape[0])
                for bbox_p in _purs_bboxes:
                    cv2.polylines(np.uint8(input_copy1), [np.reshape(bbox_p, (-1, 1, 2))], True, (255, 0, 0))

            input_copy2 = input_copy.copy()
            _tmp_bboxes = np.int32(bboxes.copy())
            _tmp_bboxes[:, :, 0] = np.clip(_tmp_bboxes[:, :, 0], 0, input.shape[1])
            _tmp_bboxes[:, :, 1] = np.clip(_tmp_bboxes[:, :, 1], 0, input.shape[0])
            for bbox in _tmp_bboxes:
                cv2.polylines(np.uint8(input_copy2), [np.reshape(bbox, (-1, 1, 2))], True, (255, 0, 0))

            region_scores_color = cv2.applyColorMap(np.uint8(region_score), cv2.COLORMAP_JET)
            region_scores_color = cv2.resize(region_scores_color, (input.shape[1], input.shape[0]))

            # viz_image2 = np.hstack([input_copy[:, :, ::-1], region_scores_color, color_markers,
            #                        input_copy1[:, :, ::-1], input_copy2[:, :, ::-1]])
            # cv2.imwrite('/nas/home/gmuffiness/result/temp_hstack.jpg', viz_image2)

            # gaussian
            target = self.gen.generate_region(region_scores_color.shape, [_tmp_bboxes])
            target_color = cv2.applyColorMap(target.astype('uint8'), cv2.COLORMAP_JET)

            overlay_img = cv2.addWeighted(input_copy[:, :, ::-1], 0.7, target_color, 0.3, 5)
            # ori img , region score, watershed, box img
            viz_image = np.hstack([input_copy[:, :, ::-1], region_scores_color, color_markers,
                                   input_copy1[:, :, ::-1], input_copy2[:, :, ::-1], target_color, overlay_img])

            save_path = os.path.join(config.RESULT_DIR, str(config.ITER // 100))
            if not os.path.exists(os.path.dirname(save_path)):
                os.makedirs(os.path.dirname(save_path))
            cv2.imwrite(os.path.join(save_path, '{}_{}'.format(img_name, 'hstack.jpg')), viz_image)
            # if config.ITER == 0:
            # cv2.imwrite(os.path.join(os.path.join(save_path, 'ori_img_v3'), '{}_{}'.format(img_name, 'img.jpg')), input_copy[:, :, ::-1])
            # cv2.imwrite(os.path.join(os.path.join(save_path, 'region_score_v3'), '{}_{}'.format(img_name, 'region_score.jpg')), bgr_region_scores)

            vis_opt = False

            # -----------------------------------------------------------------------------------------------#

        bboxes /= scale

        try:  # problem
            for k in range(len(bboxes)):
                ones = np.ones((4, 1))
                tmp = np.concatenate([bboxes[k], ones], axis=-1)
                I = np.matrix(MM).I
                ori = np.matmul(I, tmp.transpose(1, 0)).transpose(1, 0)
                bboxes[k] = ori[:, :2]

        except Exception as e:
            print(e)

        bb1 = bboxes.copy()

        if len(bboxes) > 0:
            bboxes[:, :, 1] = np.clip(bboxes[:, :, 1], 0, image.shape[0])
            bboxes[:, :, 0] = np.clip(bboxes[:, :, 0], 0, image.shape[1])

        # for cb in bboxes:
        #     # if (cb < 0).astype('float32').sum() > 0:
        #     #     import ipdb;
        #
        #     #check 1
        #     poly = plg.Polygon(cb)
        #     area = poly.area()
        #     if area < 10:
        #         import ipdb;ipdb.set_trace()
        #
        #     # check 2
        #     pursedo_bboxes_ = cb.copy()
        #     top_left = np.array([np.min(pursedo_bboxes[:, 0]), np.min(pursedo_bboxes[:, 1])]).astype(np.int32)
        #     pursedo_bboxes_ -= top_left[None, :]
        #
        #     width, height = np.max(pursedo_bboxes_[:, 0]).astype(np.int32), np.max(
        #         pursedo_bboxes_[:, 1]).astype(np.int32)
        #
        #     if width >0 or height >0:
        #         pass
        #     else:
        #         import ipdb;ipdb.set_trace()
        #         print("filter bboxes", pursedo_bboxes[i])  # 작은 box들

    if not net.training:
        net.train()

    return bboxes, region_score, confidence