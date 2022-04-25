import torch
import os
import cv2
import numpy as np
from lib.utils.render import Renderer
from PIL import ImageFont
from PIL import Image
from PIL import ImageDraw
from tqdm import trange

################# pw3d ##################
FONT_HEIGHT = 200
FONT_SIZE = 95
FRAME_RATE = 30
COLOR_GRAY = [230, 230, 230]
FONT_ORI = "(a) Video"
FONT_GT = "(b) Ground Truth"
FONT_IN = "(c) Estimator (100%)"
FONT_INMASKED = "(d) Estimator(10%)"
FONT_DECIWATCH = "(e) DeciWatch"


def visualize_smpl_pw3d(data_imgname,
                        vis_output_video_path,
                        vis_output_video_name,
                        smpl_neural,
                        in_poses,
                        gt_poses,
                        deciwatch_poses,
                        start_frame,
                        end_frame,
                        interval=10):
    video_base_path = os.path.dirname(data_imgname[0])
    print(f"You are visualizing the result of {video_base_path} ...")

    with torch.no_grad():
        in_smpl = smpl_neural(
            body_pose=torch.from_numpy(in_poses[:, 3:]).float(),
            global_orient=torch.from_numpy(in_poses[:, 0:3]).float())

        gt_smpl = smpl_neural(
            body_pose=torch.from_numpy(gt_poses[:, 3:]).float(),
            global_orient=torch.from_numpy(gt_poses[:, 0:3]).float())

        deciwatch_smpl = smpl_neural(
            body_pose=torch.from_numpy(deciwatch_poses[:, 3:]).float(),
            global_orient=torch.from_numpy(deciwatch_poses[:, 0:3]).float())

    imgsize_h, imgsize_w = cv2.imread(data_imgname[0]).shape[:2]

    videoWriter = cv2.VideoWriter(
        os.path.join(vis_output_video_path, vis_output_video_name),
        cv2.VideoWriter_fourcc(*'mp4v'), FRAME_RATE,
        (imgsize_w * 5, imgsize_h + FONT_HEIGHT))

    if not os.path.exists(vis_output_video_path):
        os.makedirs(vis_output_video_path)

    for frame_i in trange(max(0, start_frame), min(len(data_imgname),
                                                   end_frame)):
        oriimg = cv2.imread(data_imgname[frame_i])
        gt_image = np.zeros((imgsize_h, imgsize_w, 3))
        in_image = np.zeros((imgsize_h, imgsize_w, 3))
        inmasked_image = np.zeros((imgsize_h, imgsize_w, 3))
        deciwatch_image = np.zeros((imgsize_h, imgsize_w, 3))
        render = Renderer(smpl_neural.faces,
                          resolution=(imgsize_w, imgsize_h),
                          orig_img=True,
                          wireframe=False)

        gt_rendered_img = render.render(
            gt_image,
            gt_smpl.vertices.numpy()[frame_i],
            [imgsize_h * 0.001 * 0.6, imgsize_w * 0.001 * 0.6, 0.1, 0.2],
            color=np.array(COLOR_GRAY) / 255)
        in_rendered_img = render.render(
            in_image,
            in_smpl.vertices.numpy()[frame_i],
            [imgsize_h * 0.001 * 0.6, imgsize_w * 0.001 * 0.6, 0.1, 0.2],
            color=np.array(COLOR_GRAY) / 255)
        if frame_i % interval == 0:
            inmasked_rendered_img = render.render(
                inmasked_image,
                in_smpl.vertices.numpy()[frame_i],
                [imgsize_h * 0.001 * 0.6, imgsize_w * 0.001 * 0.6, 0.1, 0.2],
                color=np.array(COLOR_GRAY) / 255)
        else:
            inmasked_rendered_img = inmasked_image
        deciwatch_rendered_img = render.render(
            deciwatch_image,
            deciwatch_smpl.vertices.numpy()[frame_i],
            [imgsize_h * 0.001 * 0.6, imgsize_w * 0.001 * 0.6, 0.1, 0.2],
            color=np.array(COLOR_GRAY) / 255)

        oriimg = np.concatenate((oriimg, np.zeros(
            (FONT_HEIGHT, imgsize_w, 3))),
                                axis=0)
        gt_rendered_img = np.concatenate(
            (gt_rendered_img, np.zeros((FONT_HEIGHT, imgsize_w, 3))), axis=0)
        in_rendered_img = np.concatenate(
            (in_rendered_img, np.zeros((FONT_HEIGHT, imgsize_w, 3))), axis=0)
        inmasked_rendered_img = np.concatenate(
            (inmasked_rendered_img, np.zeros((FONT_HEIGHT, imgsize_w, 3))),
            axis=0)
        deciwatch_rendered_img = np.concatenate(
            (deciwatch_rendered_img, np.zeros((FONT_HEIGHT, imgsize_w, 3))),
            axis=0)

        #font = ImageFont.truetype('simhei',size=FONT_SIZE)
        font = ImageFont.truetype('DejaVuSansCondensed-Bold', size=FONT_SIZE)

        img_ori = Image.fromarray(oriimg.astype(np.uint8))
        img_gt = Image.fromarray(gt_rendered_img.astype(np.uint8))
        img_in = Image.fromarray(in_rendered_img.astype(np.uint8))
        img_inmasked = Image.fromarray(inmasked_rendered_img.astype(np.uint8))
        img_deciwatch = Image.fromarray(deciwatch_rendered_img.astype(
            np.uint8))

        draw_ori = ImageDraw.Draw(img_ori)
        draw_ori.text(((imgsize_w - len(FONT_ORI) * FONT_SIZE / 2) / 2,
                       imgsize_h + FONT_HEIGHT / 4),
                      FONT_ORI, (255, 255, 255),
                      font=font)
        oriimg = np.array(img_ori)

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
        in_rendered_img = np.array(img_in)

        draw_inmasked = ImageDraw.Draw(img_inmasked)
        draw_inmasked.text(
            ((imgsize_w - len(FONT_INMASKED) * FONT_SIZE / 2) / 2,
             imgsize_h + FONT_HEIGHT / 4),
            FONT_INMASKED, (255, 255, 255),
            font=font)
        inmasked_rendered_img = np.array(img_inmasked)

        draw_detected = ImageDraw.Draw(img_deciwatch)
        draw_detected.text(
            ((imgsize_w - len(FONT_DECIWATCH) * FONT_SIZE / 2) / 2,
             imgsize_h + FONT_HEIGHT / 4),
            FONT_DECIWATCH, (255, 255, 255),
            font=font)
        detected_rendered_img = np.array(img_deciwatch)
        output_img = np.concatenate(
            (oriimg, gt_rendered_img, in_rendered_img, inmasked_rendered_img,
             detected_rendered_img),
            axis=1)

        videoWriter.write(output_img)

    videoWriter.release()
    print(f"Finish! The video is stored in "+os.path.join(vis_output_video_path, vis_output_video_name))


def visualize_smpl_aist(data_imgname,
                        vis_output_video_path,
                        vis_output_video_name,
                        smpl_neural,
                        in_poses,
                        gt_poses,
                        deciwatch_poses,
                        start_frame,
                        end_frame,
                        interval=10):
    video_base_path = os.path.dirname(data_imgname[0])
    print(f"You are visualizing the result of {video_base_path} ...")
    videoCapture = cv2.VideoCapture(video_base_path+".mp4")
    success, frame = videoCapture.read()
    if not success:
        raise IOError("Can not read the video!")

    with torch.no_grad():
        in_smpl = smpl_neural(
            body_pose=torch.from_numpy(in_poses[:, 3:]).float(),
            global_orient=torch.from_numpy(in_poses[:, 0:3]).float())

        gt_smpl = smpl_neural(
            body_pose=torch.from_numpy(gt_poses[:, 3:]).float(),
            global_orient=torch.from_numpy(gt_poses[:, 0:3]).float())

        deciwatch_smpl = smpl_neural(
            body_pose=torch.from_numpy(deciwatch_poses[:, 3:]).float(),
            global_orient=torch.from_numpy(deciwatch_poses[:, 0:3]).float())

    imgsize_h, imgsize_w = frame.shape[:2]

    videoWriter = cv2.VideoWriter(
        os.path.join(vis_output_video_path, vis_output_video_name),
        cv2.VideoWriter_fourcc(*'mp4v'), FRAME_RATE,
        (imgsize_w * 5, imgsize_h + FONT_HEIGHT))

    if not os.path.exists(vis_output_video_path):
        os.makedirs(vis_output_video_path)

    for frame_i in trange(max(0, start_frame), min(len(data_imgname),
                                                   end_frame)):
        frame_number=int(os.path.basename(data_imgname[frame_i]))
        videoCapture.set(cv2.CAP_PROP_POS_FRAMES,frame_number)
        success,oriimg = videoCapture.read()
        if not success:
            raise IOError("Can not read the video!")
        gt_image = np.zeros((imgsize_h, imgsize_w, 3))
        in_image = np.zeros((imgsize_h, imgsize_w, 3))
        inmasked_image = np.zeros((imgsize_h, imgsize_w, 3))
        deciwatch_image = np.zeros((imgsize_h, imgsize_w, 3))
        render = Renderer(smpl_neural.faces,
                          resolution=(imgsize_w, imgsize_h),
                          orig_img=True,
                          wireframe=False)

        gt_rendered_img = render.render(
            gt_image,
            gt_smpl.vertices.numpy()[frame_i],
            [imgsize_h * 0.001 * 0.4, imgsize_w * 0.001 * 0.4, 0.1, 0.2],
            color=np.array(COLOR_GRAY) / 255)
        in_rendered_img = render.render(
            in_image,
            in_smpl.vertices.numpy()[frame_i],
            [imgsize_h * 0.001 * 0.4, imgsize_w * 0.001 * 0.4, 0.1, 0.2],
            color=np.array(COLOR_GRAY) / 255)
        if frame_i % interval == 0:
            inmasked_rendered_img = render.render(
                inmasked_image,
                in_smpl.vertices.numpy()[frame_i],
                [imgsize_h * 0.001 * 0.4, imgsize_w * 0.001 * 0.4, 0.1, 0.2],
                color=np.array(COLOR_GRAY) / 255)
        else:
            inmasked_rendered_img = inmasked_image
        deciwatch_rendered_img = render.render(
            deciwatch_image,
            deciwatch_smpl.vertices.numpy()[frame_i],
            [imgsize_h * 0.001 * 0.4, imgsize_w * 0.001 * 0.4, 0.1, 0.2],
            color=np.array(COLOR_GRAY) / 255)

        oriimg = np.concatenate((oriimg, np.zeros(
            (FONT_HEIGHT, imgsize_w, 3))),
                                axis=0)
        gt_rendered_img = np.concatenate(
            (gt_rendered_img, np.zeros((FONT_HEIGHT, imgsize_w, 3))), axis=0)
        in_rendered_img = np.concatenate(
            (in_rendered_img, np.zeros((FONT_HEIGHT, imgsize_w, 3))), axis=0)
        inmasked_rendered_img = np.concatenate(
            (inmasked_rendered_img, np.zeros((FONT_HEIGHT, imgsize_w, 3))),
            axis=0)
        deciwatch_rendered_img = np.concatenate(
            (deciwatch_rendered_img, np.zeros((FONT_HEIGHT, imgsize_w, 3))),
            axis=0)

        #font = ImageFont.truetype('simhei',size=FONT_SIZE)
        font = ImageFont.truetype('DejaVuSansCondensed-Bold', size=FONT_SIZE)

        img_ori = Image.fromarray(oriimg.astype(np.uint8))
        img_gt = Image.fromarray(gt_rendered_img.astype(np.uint8))
        img_in = Image.fromarray(in_rendered_img.astype(np.uint8))
        img_inmasked = Image.fromarray(inmasked_rendered_img.astype(np.uint8))
        img_deciwatch = Image.fromarray(deciwatch_rendered_img.astype(
            np.uint8))

        draw_ori = ImageDraw.Draw(img_ori)
        draw_ori.text(((imgsize_w - len(FONT_ORI) * FONT_SIZE / 2) / 2,
                       imgsize_h + FONT_HEIGHT / 4),
                      FONT_ORI, (255, 255, 255),
                      font=font)
        oriimg = np.array(img_ori)

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
        in_rendered_img = np.array(img_in)

        draw_inmasked = ImageDraw.Draw(img_inmasked)
        draw_inmasked.text(
            ((imgsize_w - len(FONT_INMASKED) * FONT_SIZE / 2) / 2,
             imgsize_h + FONT_HEIGHT / 4),
            FONT_INMASKED, (255, 255, 255),
            font=font)
        inmasked_rendered_img = np.array(img_inmasked)

        draw_detected = ImageDraw.Draw(img_deciwatch)
        draw_detected.text(
            ((imgsize_w - len(FONT_DECIWATCH) * FONT_SIZE / 2) / 2,
             imgsize_h + FONT_HEIGHT / 4),
            FONT_DECIWATCH, (255, 255, 255),
            font=font)
        detected_rendered_img = np.array(img_deciwatch)
        output_img = np.concatenate(
            (oriimg, gt_rendered_img, in_rendered_img, inmasked_rendered_img,
             detected_rendered_img),
            axis=1)

        videoWriter.write(output_img)

    videoWriter.release()
    print(f"Finish! The video is stored in "+os.path.join(vis_output_video_path, vis_output_video_name))