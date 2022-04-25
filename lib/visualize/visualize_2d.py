import os
import cv2
from PIL import ImageFont
from PIL import Image
from PIL import ImageDraw
import numpy as np
from tqdm import trange
from lib.utils.eval_metrics import *

#################### jhmdb ######################
FONT_HEIGHT = 40
FONT_SIZE = 20
JHMDB_EDGES = [
    [0, 1, 1],
    [0, 2, 1],
    [0, 3, 1],
    [3, 7, 1],
    [7, 11, 1],
    [1, 5, 1],
    [5, 9, 1],
    [9, 13, 1],
    [0, 4, 0],
    [4, 8, 0],
    [8, 12, 0],
    [1, 6, 0],
    [6, 10, 0],
    [10, 14, 0],
]

JHMDB_JOINTS = [
    [0, 0],
    [1, 1],
    [2, 1],
    [3, 1],
    [4, 0],
    [5, 1],
    [6, 0],
    [7, 1],
    [8, 0],
    [9, 1],
    [10, 0],
    [11, 1],
    [12, 0],
    [13, 1],
    [14, 0],
]

FONT_GT = "(c) Groundtruth"
FONT_DECIWATCH = "(b) Deciwatch"
FONT_IN = "(a) Simplepose"

BBOX_COLOR = (0, 0, 255)
GT_COLOR = [(0, 255, 0), (0, 255, 200)]
PRED_COLOR = [(255, 0, 0), (255, 180, 0)]
DECIWATCH_COLOR = [(0, 0, 255), (0, 180, 255)]

FRAME_RATE = 10


def plot_kpts(img, kpts, color, joints, edgs):
    for idx, lr in joints:
        joint_color = color[0] if lr else color[1]
        img = cv2.circle(
            img.copy(),
            [int(kpts[idx][0]), int(kpts[idx][1])], 2, joint_color, -1)
    for kpta, kptb, lr in edgs:
        line_color = color[0] if lr else color[1]
        img = cv2.line(img, [int(i) for i in kpts[kpta]],
                       [int(i) for i in kpts[kptb]], line_color, 1)
    return img


def plot_box(img, bbox, color=BBOX_COLOR):
    img = cv2.rectangle(img.copy(), (int(bbox[0]), int(bbox[1])),
                        (int(bbox[2]), int(bbox[3])), color, 1)
    return img


def visualize_2d_jhmdb(data_imgname,
                       vis_output_video_path,
                       vis_output_video_name,
                       predicted_pos,
                       data_pred,
                       data_gt,
                       data_bbox,
                       start_frame,
                       end_frame,
                       interval=10):
    video_base_path = os.path.dirname(data_imgname[0])
    print(f"You are visualizing the result of {video_base_path} ...")

    if not os.path.exists(vis_output_video_path):
        os.makedirs(vis_output_video_path)

    imgsize_h, imgsize_w = cv2.imread(data_imgname[0]).shape[:2]

    videoWriter = cv2.VideoWriter(
        os.path.join(vis_output_video_path, vis_output_video_name),
        cv2.VideoWriter_fourcc(*'mp4v'), FRAME_RATE,
        (imgsize_w * 3, imgsize_h + FONT_HEIGHT))

    in_pck_005=np.array(calculate_jhmdb_PCK(torch.tensor(data_pred), torch.tensor(data_gt), torch.tensor(data_bbox.astype(np.int16)), torch.tensor([1, 1]).unsqueeze(0).repeat(data_pred.shape[0],1), 0.05).mean())
    out_pck_005=np.array(calculate_jhmdb_PCK(torch.tensor(predicted_pos), torch.tensor(data_gt), torch.tensor(data_bbox.astype(np.int16)), torch.tensor([1, 1]).unsqueeze(0).repeat(data_pred.shape[0],1), 0.05).mean())

    for frame_i in trange(max(0, start_frame), min(len(data_imgname),
                                                   end_frame)):
        img = cv2.imread(data_imgname[frame_i])
        # This is for results after deciwatch
        img_deciwatch = plot_kpts(img, predicted_pos[frame_i, :],
                                  DECIWATCH_COLOR, JHMDB_JOINTS, JHMDB_EDGES)

        # This is for results before deciwatch
        img_in = plot_kpts(img, data_pred[frame_i, :], PRED_COLOR,
                           JHMDB_JOINTS, JHMDB_EDGES)
        if frame_i % interval == 0:
            img_in = plot_box(img_in, data_bbox[frame_i, :])

        img_gt = plot_kpts(img, data_gt[frame_i], GT_COLOR, JHMDB_JOINTS,
                           JHMDB_EDGES)

        img_gt = np.concatenate((img_gt, np.zeros(
            (FONT_HEIGHT, imgsize_w, 3))),
                                axis=0)
        img_in = np.concatenate((img_in, np.zeros(
            (FONT_HEIGHT, imgsize_w, 3))),
                                axis=0)
        img_deciwatch = np.concatenate(
            (img_deciwatch, np.zeros((FONT_HEIGHT, imgsize_w, 3))), axis=0)

        font = ImageFont.truetype('DejaVuSansCondensed-Bold', size=FONT_SIZE)
        #font=ImageFont.load_default()

        img_gt = Image.fromarray(img_gt.astype(np.uint8))
        img_in = Image.fromarray(img_in.astype(np.uint8))
        img_deciwatch = Image.fromarray(img_deciwatch.astype(np.uint8))

        draw_gt = ImageDraw.Draw(img_gt)
        draw_gt.text(((imgsize_w - len(FONT_GT) * FONT_SIZE / 2) / 2,
                      imgsize_h + FONT_HEIGHT / 4),
                     FONT_GT, (255, 255, 255),
                     font=font)
        gt_rendered_img = np.array(img_gt)

        draw_in = ImageDraw.Draw(img_in)
        draw_in.text(((imgsize_w - len(FONT_IN) * FONT_SIZE / 2) / 2,
                      imgsize_h + FONT_HEIGHT / 4),
                     FONT_IN, (255, 255, 255),
                     font=font)
        draw_in.text((0,0),
                     "PCK 0.05: {:0.2f}%".format(100*in_pck_005), (0, 0, 255),
                     font=font)
        in_rendered_img = np.array(img_in)

        draw_detected = ImageDraw.Draw(img_deciwatch)
        draw_detected.text(
            ((imgsize_w - len(FONT_DECIWATCH) * FONT_SIZE / 2) / 2,
             imgsize_h + FONT_HEIGHT / 4),
            FONT_DECIWATCH, (255, 255, 255),
            font=font)
        draw_detected.text((0,0),
                     "PCK 0.05: {:0.2f}%".format(100*out_pck_005), (0, 0, 255),
                     font=font)
        detected_rendered_img = np.array(img_deciwatch)
        final_img = np.concatenate(
            (in_rendered_img, detected_rendered_img, gt_rendered_img), axis=1)

        videoWriter.write(final_img)

    videoWriter.release()
    print(f"Finish! The video is stored in "+os.path.join(vis_output_video_path, vis_output_video_name))