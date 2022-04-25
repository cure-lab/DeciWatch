import os
from lib.utils.eval_metrics import calculate_mpjpe, calculate_pampjpe, calculate_accel_error
import cv2
import numpy as np
import torch
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import trange
from matplotlib.backends.backend_agg import FigureCanvasAgg
from scipy.interpolate import CubicSpline, interp1d

SUB_FIG_SIZE = 4
SUB_FIG_UNIT = 100
VIEW_NUM = 2
VIEW_CAMERA = [[10, 10], [45, 10]]

SKELETON_3D_RADIUS = 1.7
SKELETON_AX_DIST = 6.5
FRAME_RATE = 30

SKELRTON_COLOR = ['red', 'black']


def plot_skeleton(ax_3d, skeleton, color, edges, joints):
    skeleton = -skeleton[:, [2, 0, 1]]
    center = np.mean(skeleton, axis=0)

    ax_3d.set_xlim3d([
        center[0] - SKELETON_3D_RADIUS / 2, center[0] + SKELETON_3D_RADIUS / 2
    ])
    ax_3d.set_ylim3d([
        center[1] - SKELETON_3D_RADIUS / 2, center[1] + SKELETON_3D_RADIUS / 2
    ])
    ax_3d.set_zlim3d([
        center[2] - SKELETON_3D_RADIUS / 2, center[2] + SKELETON_3D_RADIUS / 2
    ])

    for idx, lr in joints:
        joint_color = color[0] if lr else color[1]
        ax_3d.plot3D(skeleton[idx, 0], skeleton[idx, 1], skeleton[idx, 2],
                     joint_color)

    for kpta, kptb, lr in edges:
        line_color = color[0] if lr else color[1]
        ax_3d.plot3D(skeleton[[kpta, kptb], 0], skeleton[[kpta, kptb], 1],
                     skeleton[[kpta, kptb], 2], line_color)

    return


################## PW3D #################

PW3D_EDGES = [
    [0, 1, 0],
    [1, 2, 0],
    [2, 8, 0],
    [3, 9, 1],
    [4, 3, 1],
    [5, 4, 1],
    [6, 7, 0],
    [7, 8, 0],
    [8, 12, 0],
    [9, 12, 1],
    [10, 9, 1],
    [11, 10, 1],
    [12, 13, 1],
]

PW3D_JOINTS = [
    [0, 0],
    [1, 0],
    [2, 0],
    [3, 1],
    [4, 1],
    [5, 1],
    [6, 0],
    [7, 0],
    [8, 0],
    [9, 1],
    [10, 1],
    [11, 1],
    [12, 1],
    [13, 1],
]

def visualize_3d_pw3d(data_imgname,
                      vis_output_video_path,
                      vis_output_video_name,
                      data_pred,
                      data_gt,
                      predicted_pos,
                      start_frame,
                      end_frame,
                      interval=10,
                      visual_interp=True):
    video_base_path = os.path.dirname(data_imgname[0])
    print(f"You are visualizing the result of {video_base_path} ...")

    if not os.path.exists(vis_output_video_path):
        os.makedirs(vis_output_video_path)

    len_seq = data_gt.shape[0]

    m2mm=1000

    # calculate errors
    mpjpe_in = np.array(calculate_mpjpe(data_pred, data_gt).cpu())*m2mm
    mpjpe_out = np.array(calculate_mpjpe(predicted_pos, data_gt).cpu())*m2mm
    pampjpe_in = np.array(calculate_pampjpe(data_pred, data_gt).cpu())*m2mm
    pampjpe_out = np.array(calculate_pampjpe(predicted_pos, data_gt).cpu())*m2mm
    acc_in = np.concatenate(
        (np.array([0]),
         np.array(calculate_accel_error(data_pred,
                                        data_gt).cpu()), np.array([0])))*m2mm
    acc_out = np.concatenate(
        (np.array([0]),
         np.array(calculate_accel_error(predicted_pos,
                                        data_gt).cpu()), np.array([0])))*m2mm


    data_pred = np.array(data_pred.cpu())
    data_gt = np.array(data_gt.cpu())
    predicted_pos = np.array(predicted_pos.cpu())

    choose_index = range(0, len_seq, interval)
    choose_mask = np.zeros((len_seq, 1, 1))  #[T, 1, 1]
    choose_mask[range(0, len_seq, interval)] = 1
    mask_data_pred = data_pred * choose_mask

    data_cubicspline = CubicSpline(choose_index,
                                   data_pred[choose_index, :, :])(range(
                                       0, len_seq))
    data_interp1d = interp1d(choose_index,
                             data_pred[choose_index, :, :].transpose(1, 2, 0),
                             kind='linear')(range(0,
                                                  len_seq)).transpose(2, 0, 1)
    mpjpe_linear = np.array(calculate_mpjpe(torch.tensor(data_interp1d), torch.tensor(data_gt)).cpu())*m2mm
    acc_linear = np.concatenate(
        (np.array([0]),
         np.array(calculate_accel_error(torch.tensor(data_interp1d),
                                        torch.tensor(data_gt)).cpu()), np.array([0])))*m2mm

    if visual_interp:
        anim_output = {
            'Estimator (Watch 100%)': data_pred,
            'Estimator (Watch 10%)': mask_data_pred,
            'Estimator (Watch 10%)+DeciWatch': predicted_pos,
            'Linear Interpolation': data_interp1d,
            'Cubic Spline Interpolation': data_cubicspline,
            'Ground truth': data_gt
        }

    else:
        anim_output = {
            'Estimator (Watch 100%)': data_pred,
            'Estimator (Watch 10%)': mask_data_pred,
            'Estimator (Watch 10%)+DeciWatch': predicted_pos,
            'Ground truth': data_gt
        }

    videoWriter = cv2.VideoWriter(
        os.path.join(vis_output_video_path, vis_output_video_name),
        cv2.VideoWriter_fourcc(*'mp4v'), FRAME_RATE,
        (SUB_FIG_SIZE * VIEW_NUM * len(anim_output) * SUB_FIG_UNIT,
         SUB_FIG_SIZE * 3 * SUB_FIG_UNIT))

    for frame_i in trange(max(0, start_frame), min(len_seq, end_frame)):
        fig = plt.figure(figsize=(SUB_FIG_SIZE * VIEW_NUM * len(anim_output),
                                  SUB_FIG_SIZE * 3))
        plt.subplots_adjust(left=None,
                            bottom=None,
                            right=None,
                            top=None,
                            wspace=None,
                            hspace=0.5)

        for view_i in range(VIEW_NUM):
            view_camera = VIEW_CAMERA[view_i]
            for index, (title, data) in enumerate(anim_output.items()):
                ax_3d = fig.add_subplot(3,
                                        len(anim_output) * VIEW_NUM,
                                        (view_i) * len(anim_output) + index +
                                        1,
                                        projection='3d')
                ax_3d.view_init(elev=view_camera[0], azim=view_camera[1])

                ax_3d.set_xticklabels([])
                ax_3d.set_yticklabels([])
                ax_3d.set_zticklabels([])

                ax_3d.dist = SKELETON_AX_DIST
                ax_3d.set_title(title, fontsize=3*SUB_FIG_SIZE)

                plot_skeleton(ax_3d, data[frame_i, :, :], SKELRTON_COLOR,
                              PW3D_EDGES, PW3D_JOINTS)

        ax_acc = fig.add_subplot(3, 1, 2)
        ax_acc.set_title('Accel Error Visualize', fontsize=4*SUB_FIG_SIZE)

        ax_mpjpe = fig.add_subplot(3, 1, 3)
        ax_mpjpe.set_title('MPJPE Visualize', fontsize=4*SUB_FIG_SIZE)

        ax_acc.plot(acc_in[:frame_i],
                    color=(202 / 255, 0 / 255, 32 / 255),
                    label='Estimator (100%) (Accel)')
        ax_acc.plot(acc_out[:frame_i],
                    color='c',
                    label='Estimator (10%)+DeciWatch (Accel)')
        ax_acc.plot(acc_linear[:frame_i],
                    color=(117/255,112/255,179/255),
                    label='Linear Interpolation')
        ax_acc.legend()
        ax_acc.grid(True)
        ax_acc.set_xlabel('Frame', fontsize=3*SUB_FIG_SIZE)
        ax_acc.set_ylabel('Mean Acceleration Error (mm/s2)', fontsize=3*SUB_FIG_SIZE)
        ax_acc.set_xlim((0, len(acc_in)))
        ax_acc.set_ylim((0, np.max((np.max(acc_in), np.max(acc_out)))))
        ax_acc.tick_params(axis="x", labelsize=3*SUB_FIG_SIZE)
        ax_acc.tick_params(axis="y", labelsize=3*SUB_FIG_SIZE)
        ax_acc.legend(fontsize=3*SUB_FIG_SIZE)

        ax_mpjpe.plot(mpjpe_in[:frame_i],
                      color=(202 / 255, 0 / 255, 32 / 255),
                      label='Estimator (100%) (MPJPE)')
        ax_mpjpe.plot(mpjpe_out[:frame_i],
                      color='c',
                      label='Estimator (10%)+DeciWatch (MPJPE)')
        ax_mpjpe.plot(mpjpe_linear[:frame_i],
                    color=(117/255,112/255,179/255),
                    label='Linear Interpolation')
        ax_mpjpe.legend()
        ax_mpjpe.grid(True)
        ax_mpjpe.set_xlabel('Frame', fontsize=3*SUB_FIG_SIZE)
        ax_mpjpe.set_ylabel('Mean Position Error (mm)', fontsize=3*SUB_FIG_SIZE)
        ax_mpjpe.set_xlim((0, len(mpjpe_in)))
        ax_mpjpe.set_ylim((0, np.max((np.max(mpjpe_in), np.max(mpjpe_out)))))
        ax_mpjpe.tick_params(axis="x", labelsize=3*SUB_FIG_SIZE)
        ax_mpjpe.tick_params(axis="y", labelsize=3*SUB_FIG_SIZE)
        ax_mpjpe.legend(fontsize=3*SUB_FIG_SIZE)

        canvas = FigureCanvasAgg(plt.gcf())
        canvas.draw()
        final_img = np.array(canvas.renderer.buffer_rgba())[:, :, [2, 1, 0]]

        #plt.savefig("tmp" + str(frame_i) + ".png")

        videoWriter.write(final_img)
        plt.close()

    videoWriter.release()
    print(f"Finish! The video is stored in "+os.path.join(vis_output_video_path, vis_output_video_name))




################## H36M #################

H36M_EDGES = [
    [0,1,1],
    [1,2,1],
    [2,3,1],
    [0,7,0],
    [7,8,0],
    [8,9,0],
    [9,10,0],
    [8,11,0],
    [11,12,0],
    [12,13,0],
    [0,4,0],
    [4,5,0],
    [5,6,0],
    [8,14,1],
    [14,15,1],
    [15,16,1],
]

H36M_JOINTS = [
    [0, 0],
    [1, 1],
    [2, 1],
    [3, 1],
    [4, 0],
    [5, 0],
    [6, 0],
    [7, 0],
    [8, 0],
    [9, 0],
    [10, 0],
    [11, 0],
    [12, 0],
    [13, 0],
    [14, 1],
    [15, 1],
    [16, 1],
]



def visualize_3d_h36m(data_imgname,
                      vis_output_video_path,
                      vis_output_video_name,
                      data_pred,
                      data_gt,
                      predicted_pos,
                      start_frame,
                      end_frame,
                      interval=10,
                      visual_interp=True):
    video_base_path = os.path.dirname(data_imgname[0])
    print(f"You are visualizing the result of {video_base_path} ...")

    if not os.path.exists(vis_output_video_path):
        os.makedirs(vis_output_video_path)

    len_seq = data_gt.shape[0]

    m2mm=1000

    # calculate errors
    mpjpe_in = np.array(calculate_mpjpe(data_pred, data_gt).cpu())*m2mm
    mpjpe_out = np.array(calculate_mpjpe(predicted_pos, data_gt).cpu())*m2mm
    pampjpe_in = np.array(calculate_pampjpe(data_pred, data_gt).cpu())*m2mm
    pampjpe_out = np.array(calculate_pampjpe(predicted_pos, data_gt).cpu())*m2mm
    acc_in = np.concatenate(
        (np.array([0]),
         np.array(calculate_accel_error(data_pred,
                                        data_gt).cpu()), np.array([0])))*m2mm
    acc_out = np.concatenate(
        (np.array([0]),
         np.array(calculate_accel_error(predicted_pos,
                                        data_gt).cpu()), np.array([0])))*m2mm

    data_pred = np.array(data_pred.cpu())
    data_gt = np.array(data_gt.cpu())
    predicted_pos = np.array(predicted_pos.cpu())

    choose_index = range(0, len_seq, interval)
    choose_mask = np.zeros((len_seq, 1, 1))  #[T, 1, 1]
    choose_mask[range(0, len_seq, interval)] = 1
    mask_data_pred = data_pred * choose_mask

    data_cubicspline = CubicSpline(choose_index,
                                   data_pred[choose_index, :, :])(range(
                                       0, len_seq))
    data_interp1d = interp1d(choose_index,
                             data_pred[choose_index, :, :].transpose(1, 2, 0),
                             kind='linear')(range(0,
                                                  len_seq)).transpose(2, 0, 1)
    mpjpe_linear = np.array(calculate_mpjpe(torch.tensor(data_interp1d), torch.tensor(data_gt)).cpu())*m2mm
    acc_linear = np.concatenate(
        (np.array([0]),
         np.array(calculate_accel_error(torch.tensor(data_interp1d),
                                        torch.tensor(data_gt)).cpu()), np.array([0])))*m2mm


    if visual_interp:
        anim_output = {
            'Estimator (Watch 100%)': data_pred,
            'Estimator (Watch 10%)': mask_data_pred,
            'Estimator (Watch 10%)+DeciWatch': predicted_pos,
            'Linear Interpolation': data_interp1d,
            'Cubic Spline Interpolation': data_cubicspline,
            'Ground truth': data_gt
        }

    else:
        anim_output = {
            'Estimator (Watch 100%)': data_pred,
            'Estimator (Watch 10%)': mask_data_pred,
            'Estimator (Watch 10%)+DeciWatch': predicted_pos,
            'Ground truth': data_gt
        }

    videoWriter = cv2.VideoWriter(
        os.path.join(vis_output_video_path, vis_output_video_name),
        cv2.VideoWriter_fourcc(*'mp4v'), FRAME_RATE,
        (SUB_FIG_SIZE * VIEW_NUM * len(anim_output) * SUB_FIG_UNIT,
         SUB_FIG_SIZE * 3 * SUB_FIG_UNIT))

    for frame_i in trange(max(0, start_frame), min(len_seq, end_frame)):
        fig = plt.figure(figsize=(SUB_FIG_SIZE * VIEW_NUM * len(anim_output),
                                  SUB_FIG_SIZE * 3))
        plt.subplots_adjust(left=None,
                            bottom=None,
                            right=None,
                            top=None,
                            wspace=None,
                            hspace=0.5)

        for view_i in range(VIEW_NUM):
            view_camera = VIEW_CAMERA[view_i]
            for index, (title, data) in enumerate(anim_output.items()):
                ax_3d = fig.add_subplot(3,
                                        len(anim_output) * VIEW_NUM,
                                        (view_i) * len(anim_output) + index +
                                        1,
                                        projection='3d')
                ax_3d.view_init(elev=view_camera[0], azim=view_camera[1])

                ax_3d.set_xticklabels([])
                ax_3d.set_yticklabels([])
                ax_3d.set_zticklabels([])

                ax_3d.dist = SKELETON_AX_DIST
                ax_3d.set_title(title, fontsize=3*SUB_FIG_SIZE)

                plot_skeleton(ax_3d, data[frame_i, :, :], SKELRTON_COLOR,
                              H36M_EDGES, H36M_JOINTS)

        ax_acc = fig.add_subplot(3, 1, 2)
        ax_acc.set_title('Accel Error Visualize', fontsize=4*SUB_FIG_SIZE)

        ax_mpjpe = fig.add_subplot(3, 1, 3)
        ax_mpjpe.set_title('MPJPE Visualize', fontsize=4*SUB_FIG_SIZE)

        ax_acc.plot(acc_in[:frame_i],
                    color=(202 / 255, 0 / 255, 32 / 255),
                    label='Estimator (100%) (Accel)')
        ax_acc.plot(acc_out[:frame_i],
                    'c',
                    label='Estimator (10%)+DeciWatch (Accel)')
        ax_acc.plot(acc_linear[:frame_i],
                    color=(117/255,112/255,179/255),
                    label='Linear Interpolation')
        ax_acc.legend()
        ax_acc.grid(True)
        ax_acc.set_xlabel('Frame', fontsize=3*SUB_FIG_SIZE)
        ax_acc.set_ylabel('Mean Acceleration Error (mm/s2)', fontsize=3*SUB_FIG_SIZE)
        ax_acc.set_xlim((0, len(acc_in)))
        ax_acc.set_ylim((0, np.max((np.max(acc_in), np.max(acc_out)))))
        ax_acc.tick_params(axis="x", labelsize=3*SUB_FIG_SIZE)
        ax_acc.tick_params(axis="y", labelsize=3*SUB_FIG_SIZE)
        ax_acc.legend(fontsize=3*SUB_FIG_SIZE)

        ax_mpjpe.plot(mpjpe_in[:frame_i],
                      color=(202 / 255, 0 / 255, 32 / 255),
                      label='Estimator (100%) (MPJPE)')
        ax_mpjpe.plot(mpjpe_out[:frame_i],
                      'c',
                      label='Estimator (10%)+DeciWatch (MPJPE)')
        ax_mpjpe.plot(mpjpe_linear[:frame_i],
                    color=(117/255,112/255,179/255),
                    label='Linear Interpolation')
        ax_mpjpe.legend()
        ax_mpjpe.grid(True)
        ax_mpjpe.set_xlabel('Frame', fontsize=3*SUB_FIG_SIZE)
        ax_mpjpe.set_ylabel('Mean Position Error (mm)', fontsize=3*SUB_FIG_SIZE)
        ax_mpjpe.set_xlim((0, len(mpjpe_in)))
        ax_mpjpe.set_ylim((0, np.max((np.max(mpjpe_in), np.max(mpjpe_out)))))
        ax_mpjpe.tick_params(axis="x", labelsize=3*SUB_FIG_SIZE)
        ax_mpjpe.tick_params(axis="y", labelsize=3*SUB_FIG_SIZE)
        ax_mpjpe.legend(fontsize=3*SUB_FIG_SIZE)

        canvas = FigureCanvasAgg(plt.gcf())
        canvas.draw()
        final_img = np.array(canvas.renderer.buffer_rgba())[:, :, [2, 1, 0]]

        #plt.savefig("tmp" + str(frame_i) + ".png")

        videoWriter.write(final_img)
        plt.close()

    videoWriter.release()
    print(f"Finish! The video is stored in "+os.path.join(vis_output_video_path, vis_output_video_name))


################## AIST #################

AIST_EDGES = [
    [0, 1, 0],
    [1, 2, 0],
    [2, 8, 0],
    [3, 9, 1],
    [4, 3, 1],
    [5, 4, 1],
    [6, 7, 0],
    [7, 8, 0],
    [8, 12, 0],
    [9, 12, 1],
    [10, 9, 1],
    [11, 10, 1],
    [12, 13, 1],
]

AIST_JOINTS = [
    [0, 0],
    [1, 0],
    [2, 0],
    [3, 1],
    [4, 1],
    [5, 1],
    [6, 0],
    [7, 0],
    [8, 0],
    [9, 1],
    [10, 1],
    [11, 1],
    [12, 1],
    [13, 1],
]



def visualize_3d_aist(data_imgname,
                      vis_output_video_path,
                      vis_output_video_name,
                      data_pred,
                      data_gt,
                      predicted_pos,
                      start_frame,
                      end_frame,
                      interval=10,
                      visual_interp=True):
    video_base_path = os.path.dirname(data_imgname[0])
    print(f"You are visualizing the result of {video_base_path} ...")

    if not os.path.exists(vis_output_video_path):
        os.makedirs(vis_output_video_path)

    len_seq = data_gt.shape[0]

    m2mm=1000

    # calculate errors
    mpjpe_in = np.array(calculate_mpjpe(data_pred, data_gt).cpu())*m2mm
    mpjpe_out = np.array(calculate_mpjpe(predicted_pos, data_gt).cpu())*m2mm
    pampjpe_in = np.array(calculate_pampjpe(data_pred, data_gt).cpu())*m2mm
    pampjpe_out = np.array(calculate_pampjpe(predicted_pos, data_gt).cpu())*m2mm
    acc_in = np.concatenate(
        (np.array([0]),
         np.array(calculate_accel_error(data_pred,
                                        data_gt).cpu()), np.array([0])))*m2mm
    acc_out = np.concatenate(
        (np.array([0]),
         np.array(calculate_accel_error(predicted_pos,
                                        data_gt).cpu()), np.array([0])))*m2mm

    data_pred = np.array(data_pred.cpu())
    data_gt = np.array(data_gt.cpu())
    predicted_pos = np.array(predicted_pos.cpu())

    choose_index = range(0, len_seq, interval)
    choose_mask = np.zeros((len_seq, 1, 1))  #[T, 1, 1]
    choose_mask[range(0, len_seq, interval)] = 1
    mask_data_pred = data_pred * choose_mask

    data_cubicspline = CubicSpline(choose_index,
                                   data_pred[choose_index, :, :])(range(
                                       0, len_seq))
    data_interp1d = interp1d(choose_index,
                             data_pred[choose_index, :, :].transpose(1, 2, 0),
                             kind='linear')(range(0,
                                                  len_seq)).transpose(2, 0, 1)
    mpjpe_linear = np.array(calculate_mpjpe(torch.tensor(data_interp1d), torch.tensor(data_gt)).cpu())*m2mm
    acc_linear = np.concatenate(
        (np.array([0]),
         np.array(calculate_accel_error(torch.tensor(data_interp1d),
                                        torch.tensor(data_gt)).cpu()), np.array([0])))*m2mm


    if visual_interp:
        anim_output = {
            'Estimator (Watch 100%)': data_pred,
            'Estimator (Watch 10%)': mask_data_pred,
            'Estimator (Watch 10%)+DeciWatch': predicted_pos,
            'Linear Interpolation': data_interp1d,
            'Cubic Spline Interpolation': data_cubicspline,
            'Ground truth': data_gt
        }

    else:
        anim_output = {
            'Estimator (Watch 100%)': data_pred,
            'Estimator (Watch 10%)': mask_data_pred,
            'Estimator (Watch 10%)+DeciWatch': predicted_pos,
            'Ground truth': data_gt
        }

    videoWriter = cv2.VideoWriter(
        os.path.join(vis_output_video_path, vis_output_video_name),
        cv2.VideoWriter_fourcc(*'mp4v'), FRAME_RATE,
        (SUB_FIG_SIZE * VIEW_NUM * len(anim_output) * SUB_FIG_UNIT,
         SUB_FIG_SIZE * 3 * SUB_FIG_UNIT))

    for frame_i in trange(max(0, start_frame), min(len_seq, end_frame)):
        fig = plt.figure(figsize=(SUB_FIG_SIZE * VIEW_NUM * len(anim_output),
                                  SUB_FIG_SIZE * 3))
        plt.subplots_adjust(left=None,
                            bottom=None,
                            right=None,
                            top=None,
                            wspace=None,
                            hspace=0.5)

        for view_i in range(VIEW_NUM):
            view_camera = VIEW_CAMERA[view_i]
            for index, (title, data) in enumerate(anim_output.items()):
                ax_3d = fig.add_subplot(3,
                                        len(anim_output) * VIEW_NUM,
                                        (view_i) * len(anim_output) + index +
                                        1,
                                        projection='3d')
                ax_3d.view_init(elev=view_camera[0], azim=view_camera[1])

                ax_3d.set_xticklabels([])
                ax_3d.set_yticklabels([])
                ax_3d.set_zticklabels([])

                ax_3d.dist = SKELETON_AX_DIST
                ax_3d.set_title(title, fontsize=3*SUB_FIG_SIZE)

                plot_skeleton(ax_3d, data[frame_i, :, :], SKELRTON_COLOR,
                              AIST_EDGES, AIST_JOINTS)

        ax_acc = fig.add_subplot(3, 1, 2)
        ax_acc.set_title('Accel Error Visualize', fontsize=4*SUB_FIG_SIZE)

        ax_mpjpe = fig.add_subplot(3, 1, 3)
        ax_mpjpe.set_title('MPJPE Visualize', fontsize=4*SUB_FIG_SIZE)

        ax_acc.plot(acc_in[:frame_i],
                    color=(202 / 255, 0 / 255, 32 / 255),
                    label='Estimator (100%) (Accel)')
        ax_acc.plot(acc_out[:frame_i],
                    'c',
                    label='Estimator (10%)+DeciWatch (Accel)')
        ax_acc.plot(acc_linear[:frame_i],
                    color=(117/255,112/255,179/255),
                    label='Linear Interpolation')
        ax_acc.legend()
        ax_acc.grid(True)
        ax_acc.set_xlabel('Frame', fontsize=3*SUB_FIG_SIZE)
        ax_acc.set_ylabel('Mean Acceleration Error (mm/s2)', fontsize=3*SUB_FIG_SIZE)
        ax_acc.set_xlim((0, len(acc_in)))
        ax_acc.set_ylim((0, np.max((np.max(acc_in), np.max(acc_out)))))
        ax_acc.tick_params(axis="x", labelsize=3*SUB_FIG_SIZE)
        ax_acc.tick_params(axis="y", labelsize=3*SUB_FIG_SIZE)
        ax_acc.legend(fontsize=3*SUB_FIG_SIZE)

        ax_mpjpe.plot(mpjpe_in[:frame_i],
                      color=(202 / 255, 0 / 255, 32 / 255),
                      label='Estimator (100%) (MPJPE)')
        ax_mpjpe.plot(mpjpe_out[:frame_i],
                      'c',
                      label='Estimator (10%)+DeciWatch (MPJPE)')
        ax_mpjpe.plot(mpjpe_linear[:frame_i],
                    color=(117/255,112/255,179/255),
                    label='Linear Interpolation')
        ax_mpjpe.legend()
        ax_mpjpe.grid(True)
        ax_mpjpe.set_xlabel('Frame', fontsize=3*SUB_FIG_SIZE)
        ax_mpjpe.set_ylabel('Mean Position Error (mm)', fontsize=3*SUB_FIG_SIZE)
        ax_mpjpe.set_xlim((0, len(mpjpe_in)))
        ax_mpjpe.set_ylim((0, np.max((np.max(mpjpe_in), np.max(mpjpe_out)))))
        ax_mpjpe.tick_params(axis="x", labelsize=3*SUB_FIG_SIZE)
        ax_mpjpe.tick_params(axis="y", labelsize=3*SUB_FIG_SIZE)
        ax_mpjpe.legend(fontsize=3*SUB_FIG_SIZE)

        canvas = FigureCanvasAgg(plt.gcf())
        canvas.draw()
        final_img = np.array(canvas.renderer.buffer_rgba())[:, :, [2, 1, 0]]

        #plt.savefig("tmp" + str(frame_i) + ".png")

        videoWriter.write(final_img)
        plt.close()

    videoWriter.release()
    print(f"Finish! The video is stored in "+os.path.join(vis_output_video_path, vis_output_video_name))




