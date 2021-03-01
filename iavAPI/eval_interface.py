import cv2
from layers.output_utils import postprocess, undo_image_transformation
from utils import timer
import torch
from collections import defaultdict
from data import cfg, COLORS

from scipy.spatial.distance import cdist
import numpy as np

color_cache = defaultdict(lambda: {})

from iavAPI.utils import calculations

degree_sign= u'\N{DEGREE SIGN}'

def prep_display(dets_out, img, h, w,args, undo_transform=True, class_color=False, mask_alpha=0.45, fps_str=''):
    """
    Note: If undo_transform=False then im_h and im_w are allowed to be None.
    """
    #ToDo Eigene Settings Klasse die EIngabe liest und sortiert
    FLAG_Default_Detect_ROT_BBOX =True
    FLAG_Fix_Color_Images =True
    if args.IAV_analyze:
        args.subtract_background = True
        args.binarize = True
        args.display_fps = False
        args.display_fps_eval = True
        args.display_bboxes= False
        args.display_text=False
        FLAG_Default_Detect_ROT_BBOX = False

    elif FLAG_Default_Detect_ROT_BBOX:
        args.subtract_background = False
        args.binarize = True
        args.display_fps_eval = True
        args.display_bboxes = False
        args.display_text = True

    else:
        args.subtract_background = False
        args.binarize = False
        args.display_fps_eval = False
        args.display_bboxes = True
        args.display_text = True
        FLAG_Default_Detect_ROT_BBOX = False





    if undo_transform:
        img_numpy = undo_image_transformation(img, w, h)
        img_gpu = torch.Tensor(img_numpy).cuda()
    else:
        img_gpu = img / 255.0
        h, w, _ = img.shape

    with timer.env('Postprocess'):
        save = cfg.rescore_bbox
        cfg.rescore_bbox = True
        t = postprocess(dets_out, w, h, visualize_lincomb=args.display_lincomb,
                        crop_masks=args.crop,
                        score_threshold=args.score_threshold)
        cfg.rescore_bbox = save

    with timer.env('Copy'):
        idx = t[1].argsort(0, descending=True)[:args.top_k]
        if cfg.eval_mask_branch:
            # Masks are drawn on the GPU, so don't copy
            masks = t[3][idx]
        classes, scores, boxes = [x[idx].cpu().numpy() for x in t[:3]]

    num_dets_to_consider = min(args.top_k, classes.shape[0])
    for j in range(num_dets_to_consider):
        if scores[j] < args.score_threshold:
            num_dets_to_consider = j
            break

    # Quick and dirty lambda for selecting the color for a particular index
    # Also keeps track of a per-gpu color cache for maximum speed
    def get_color(j, on_gpu=None):
        global color_cache
        color_idx = (classes[j] * 5 if class_color else j * 5) % len(COLORS)

        if on_gpu is not None and color_idx in color_cache[on_gpu]:
            return color_cache[on_gpu][color_idx]
        else:
            color = COLORS[color_idx]
            if not undo_transform:
                # The image might come in as RGB or BRG, depending
                color = (color[2], color[1], color[0])
            if on_gpu is not None:
                color = torch.Tensor(color).to(on_gpu).float() / 255.
                color_cache[on_gpu][color_idx] = color
            return color
    #

    if args.subtract_background:
        if cfg.eval_mask_branch and num_dets_to_consider > 0 :
            # assert not args.binarize, exit();
            # After this, mask is of size [num_dets, h, w, 1]
            masks1 = masks[:num_dets_to_consider, :, :, None]
            img_gpu_back = img_gpu * (masks1.sum(dim=0) >= 1).float().expand(-1, -1, 3)
        else:
            img_gpu_back = img_gpu * 0

    if args.binarize:
        if cfg.eval_mask_branch and num_dets_to_consider > 0:
            # assert not args.subtract_background, exit()
            # After this, mask is of size [num_dets, h, w, 1]
            masks2 = masks[:num_dets_to_consider, :, :, None]
            img_gpu_bi = (masks2.sum(dim=0) >= 1).float().expand(-1, -1, 3).contiguous()
        else:
            img_gpu_bi = img_gpu * 0

    # First, draw the masks on the GPU where we can do it really fast
    # Beware: very fast but possibly unintelligible mask-drawing code ahead
    # I wish I had access to OpenGL or Vulkan but alas, I guess Pytorch tensor operations will have to suffice
    if args.display_masks and cfg.eval_mask_branch and num_dets_to_consider > 0 and not args.IAV_analyze:
        # After this, mask is of size [num_dets, h, w, 1]
        masks = masks[:num_dets_to_consider, :, :, None]

        # Prepare the RGB images for each mask given their color (size [num_dets, h, w, 1])
        colors = torch.cat(
            [get_color(j, on_gpu=img_gpu.device.index).view(1, 1, 1, 3) for j in range(num_dets_to_consider)],
            dim=0)
        masks_color = masks.repeat(1, 1, 1, 3) * colors * mask_alpha

        # This is 1 everywhere except for 1-mask_alpha where the mask is
        inv_alph_masks = masks * (-mask_alpha) + 1

        # I did the math for this on pen and paper. This whole block should be equivalent to:
        #    for j in range(num_dets_to_consider):
        #        img_gpu = img_gpu * inv_alph_masks[j] + masks_color[j]
        masks_color_summand = masks_color[0]
        if num_dets_to_consider > 1:
            inv_alph_cumul = inv_alph_masks[:(num_dets_to_consider - 1)].cumprod(dim=0)
            masks_color_cumul = masks_color[1:] * inv_alph_cumul
            masks_color_summand += masks_color_cumul.sum(dim=0)

        img_gpu = img_gpu * inv_alph_masks.prod(dim=0) + masks_color_summand

    if args.display_fps:
        # Draw the box for the fps on the GPU
        font_face = cv2.FONT_HERSHEY_DUPLEX
        font_scale = 0.6
        font_thickness = 1

        text_w, text_h = cv2.getTextSize(fps_str, font_face, font_scale, font_thickness)[0]

        img_gpu[0:text_h + 8, 0:text_w + 8] *= 0.6  # 1 - Box alpha

    # Then draw the stuff that needs to be done on the cpu
    # Note, make sure this is a uint8 tensor or opencv will not anti alias text for whatever reason


    if args.subtract_background:
        img_numpy_back= (img_gpu_back * 255).byte().cpu().numpy()

    if args.binarize:
        img_numpy_binary = (img_gpu_bi * 255).byte().cpu().numpy()
        if FLAG_Default_Detect_ROT_BBOX:
            img_numpy = (img_gpu * 255).byte().cpu().numpy()


    else:
        img_numpy =  (img_gpu * 255).byte().cpu().numpy()



    if args.display_fps:
        # Draw the text on the CPU
        text_pt = (4, text_h + 2)
        text_color = [255, 255, 255]

        cv2.putText(img_numpy, fps_str, text_pt, font_face, font_scale, text_color, font_thickness, cv2.LINE_AA)

        if args.log_fps:
            global fps_list
            fps_list.append(fps_str)


    if num_dets_to_consider == 0:
        if args.subtract_background:
            return img_numpy_back
        if args.binarize:
            return img_numpy_binary

        if FLAG_Default_Detect_ROT_BBOX:
            return img_numpy

        else:
            return img_numpy



    if args.display_text or args.display_bboxes:
        center = []
        color_list= []
        for j in reversed(range(num_dets_to_consider)):
            x1, y1, x2, y2 = boxes[j, :]
            color = get_color(j)
            score = scores[j]

            if args.display_bboxes:
                cv2.rectangle(img_numpy, (x1, y1), (x2, y2), color, 2)


            if args.display_text:
                _class = cfg.dataset.class_names[classes[j]]
                text_str = '%s: %.2f' % (_class, score) if args.display_scores else _class

                font_face = cv2.FONT_HERSHEY_DUPLEX
                font_scale = 0.6
                font_thickness = 1

                text_w, text_h = cv2.getTextSize(text_str, font_face, font_scale, font_thickness)[0]

                text_pt = (x1, y1 - 3)
                text_color = [255, 255, 255]

                cv2.rectangle(img_numpy, (x1, y1), (x1 + text_w, y1 - text_h - 4), color, -1)
                cv2.putText(img_numpy, text_str, text_pt, font_face, font_scale, text_color, font_thickness,
                            cv2.LINE_AA)

            if FLAG_Fix_Color_Images:
                xCenter = int((x1+x2)/2)
                yCenter = int((y1 + y2)/2)
                center.append((xCenter,yCenter))
                color_list.append(color)

        if FLAG_Default_Detect_ROT_BBOX:
            rot_bboxes = calculations.calc_rotated_bbox_full(img_numpy_binary, slice=True,
                                                           contour_mode=cv2.CHAIN_APPROX_NONE)
            if rot_bboxes is not None:
                if FLAG_Fix_Color_Images:
                    ordered_bboxes = [calculations.order_rect(rot_bbox) for rot_bbox in rot_bboxes]
                    centroids = [calculations.find_rect_centroid(top_left=(ordered_bbox[0][0], ordered_bbox[0][1]),
                                                                 bottom_right=(ordered_bbox[3][0], ordered_bbox[3][1]))
                                 for ordered_bbox in ordered_bboxes]


                    minima_indices = calculations.sort_by_distance(center,centroids, returnType='minima_indices')
                    bboxes_aligned = [rot_bboxes[indice] for indice in minima_indices]

                    if rot_bboxes is not None:
                        [cv2.drawContours(img_numpy, [rot_bbox], 0, color_list[colr_idx], 2,lineType=cv2.LINE_AA) for colr_idx, rot_bbox in
                         enumerate(bboxes_aligned)]

                else:
                    [cv2.drawContours(img_numpy, [rot_bbox], 0, (54, 67, 244), 2) for rot_bbox in rot_bboxes]



    if args.IAV_analyze:
        img_numpy = analyze(img_numpy_binary,img_numpy_back,
                            draw_rot_bbox=False, draw_Orientation=True,draw_keypoints=False,draw_dist=False)



    if args.display_fps_eval:
        font_face = cv2.FONT_HERSHEY_DUPLEX
        font_scale = 0.6
        font_thickness = 1

        text_pt = (4, 14 + 2)
        text_color = [255, 255, 255]

        cv2.putText(img_numpy, fps_str, text_pt, font_face, font_scale, text_color, font_thickness, cv2.LINE_AA)


    return img_numpy



def analyze(img_array,img_array_back=None,draw_rot_bbox=False, draw_Orientation=False,draw_keypoints=True,draw_dist=True):

    cntrs = calculations.getContours(img_array, mode=cv2.CHAIN_APPROX_NONE)

    if len(cntrs) == 0 or cntrs is None:
        return img_array_back

    try: #ToDO add if cntrs is not None or weird Bug with slice step cannot be zero
        cntrs = calculations.sliceNSortContours(cntrs, slice_stride=30, sort_mode=True)
    except:
        pass
    number_detectons = len(cntrs)
    dist_cache = []
    for idx in range(number_detectons):
        # if idx < number_detectons-1:
        if draw_dist:
            pairwise_min_distances = np.array([calculations.calc_min_dist(cntrs[idx], cnt) for cdx, cnt in enumerate(cntrs) if cdx is not idx], dtype='object')
            if len(pairwise_min_distances) > 0:
                abs_min_idx = np.argmin(pairwise_min_distances[:, 1][:])
                closest_points, dist = pairwise_min_distances[abs_min_idx]
                p1, p2 = tuple(closest_points[0]), tuple(closest_points[1])
                # p1, p2, dist = calculations.calc_min_dist(cntrs[idx], cntrs[idx+1])
                # dist_cache.append((p1,p2,dist))
                dist_str = str(round(dist,1))+'px'
                font_scale = 0.35
                text_center, text_width, text_height = calculations.getTextTag((int(((p1[0]+p2[0])/2)),
                                                                    int(((p1[1]+p2[1])/2))), dist_str,
                                                                               fontScale=font_scale)

                cv2.line(img_array_back, p1, p2, COLORS[idx], 2,lineType=cv2.LINE_AA)

                cv2.rectangle(img_array_back, text_center, (text_center[0] + text_width, text_center[1] - text_height - 4),
                              COLORS[idx], -1)

                cv2.putText(img_array_back, dist_str, (int(text_center[0]), int(text_center[1])),
                            cv2.FONT_HERSHEY_DUPLEX,0.35, (255,255,255), 1, cv2.LINE_AA)



        if draw_Orientation:
            bbox, center, angle, width, height = calculations.calc_rotated_bbox(cntrs[idx])
            axis_long, axis_short = calculations.getOrientationAxis(center, angle, height, width, scale=0.75)
            # calculations.fit_ellipse(cntrs[idx])

            cv2.arrowedLine(img_array_back, pt1= axis_long[0], pt2= axis_long[1], color=(0,0,255), thickness=2,line_type=cv2.LINE_AA, tipLength=0.04)
            cv2.arrowedLine(img_array_back, pt1=axis_short[0], pt2=axis_short[1], color=(0, 255, 0), thickness=2,line_type=cv2.LINE_AA,tipLength=0.06)

            angle_str = f'%.1f' % angle

            font_scale = 0.4
            text_center, text_width, text_height = calculations.getTextTag(center, angle_str, fontScale=font_scale)

            cv2.rectangle(img_array_back, text_center, (text_center[0] + text_width, text_center[1] - text_height - 4), COLORS[idx], -1)

            cv2.putText(img_array_back, angle_str, (int(text_center[0]), int(text_center[1])), cv2.FONT_HERSHEY_DUPLEX,
                        font_scale, (255,255,255), 1, cv2.LINE_AA)
            cv2.circle(img_array_back, center, 3, (0,0,0), -1)

        if draw_rot_bbox:
            if draw_Orientation:

                #ToDO How to draw Orientation beautiful
                #https://www.google.com/search?q
                # =object+detection+and+classification+with
                # +rotated+bounding+boxes&rlz=1C1CHBF_deDE918DE918&sxsrf=ALeKk00C
                # SRd3cpodZrv-mryPmK2Z30drig:1613516720416&source=lnms&tbm=isch&sa=X&ved=2ahUKEwiShdaXwu_uAhXCGu
                # wKHUceBYcQ_AUoAXoECAcQAw&biw=1920&bih=937#imgrc=ankEZ5xKgWJyNM&imgdii=6FVcVUkN7yYTKM


                # So to not call calc_rotated_bbox twice
                cv2.drawContours(img_array_back, [bbox], 0, COLORS[idx], 2,lineType=cv2.LINE_AA)
            else:
                bbox = calculations.calc_rotated_bbox(cntrs[idx],mode='only_box')
                cv2.drawContours(img_array_back, [bbox], 0, COLORS[idx], 2)

        if draw_keypoints:
            [cv2.circle(img_array_back, tuple(c), 2, COLORS[idx], -1,lineType=cv2.LINE_AA) for c in cntrs[idx]]

            # Only Minima
            # color = random.choice(COLORS)
            # min_dists = [x[2] for x in dist_cache]
            # idx = np.argmin(min_dists)
            # points = dist_cache[idx][0:2]
            # cv2.line(img_array_back,points[0], points[1], color, 2)
            # cv2.putText(img_array_back, str(round(min_dists[idx], 2)) + 'px',
            #             (int(((points[0][0] + points[1][0]) / 2)), int(((points[0][1] + points[0][1]) / 2))), cv2.FONT_HERSHEY_DUPLEX, 0.4, color, 1,
            #             cv2.LINE_AA)



    # except:
    #     print('Skipped Frame')


    return img_array_back








# def _getOrientationC(contours):
#     box = np.int0(cv2.boxPoints(rect))
#
#     center = (int(rect[0][0]), int(rect[0][1]))
#     (width, height), angle = rect[1:3]
#     if width < height:
#         angle = 90 - angle
#     else:
#         angle = -angle
#
#     X_Axis = _getOrientationAxis_X(box,center,width,height)
#
#     return X_Axis, angle
#
#
# def _getOrientationAxis_X(rectPoints,center,width,height):
#     left_bottom = rectPoints[0].reshape(1,2)
#     rest_points = rectPoints[1:]
#     dists = cdist(left_bottom, rest_points)
#     clostest = rest_points[np.argmin(dists)]
#
#     bottom_middle = tuple(((left_bottom + clostest) / 2).astype(np.int0)[0])
#     return [center,bottom_middle]
#
#
# # def convex_hull_image(data):
#     # region = np.argwhere(data)
#     # hull = ConvexHull(region)
#     # verts = [(region[v,0], region[v,1]) for v in hull.vertices]
#     # img = Image.new('L', data.shape, 0)
#     # ImageDraw.Draw(img).polygon(verts, outline=1, fill=1)
#     # mask = np.array(img)
#     #
#     # return mask.T
#
# def rotated_bbox(cntrs):
#     rect = cv2.minAreaRect(cntrs)
#     box = cv2.boxPoints(rect)
#
#     return [np.int0(box)]
#
# def closest_node(node, nodes):
#     closest_index = cdist(node, nodes).argmin()
#     return nodes[closest_index]
#
# def approx_line(img=None,cntrs=None):
#     rows, cols = img.shape[:2]
#     [vx, vy, x, y] = cv2.fitLine(cntrs[0], cv2.DIST_L2, 0, 0.01, 0.01)
#     lefty = int((-x * vy / vx) + y)
#     righty = int(((cols - x) * vy / vx) + y)
#     cv2.line(img, (cols - 1, righty), (0, lefty), (0, 255, 0), 2)
# #
# def calc_min_dist(A, B):
#     distance = cdist(A, B)
#     rel_dist_minima = distance.min(axis=1)
#     abs_dist_min = rel_dist_minima.min()
#     abs_dist_min_idx = np.argmin(rel_dist_minima)
#     min_idx = np.argmin(distance, axis=1)[abs_dist_min_idx]
#
#     return tuple(A[abs_dist_min_idx]), tuple(B[min_idx]), abs_dist_min







