import torch
from lib.utils.eval_metrics import *
from lib.utils.geometry_utils import *
import os
import cv2
from lib.visualize.visualize_2d import visualize_2d_jhmdb
from lib.visualize.visualize_smpl import visualize_smpl_pw3d,visualize_smpl_aist
from lib.visualize.visualize_3d import visualize_3d_pw3d,visualize_3d_h36m,visualize_3d_aist
import sys


class Visualize():

    def __init__(self,test_dataset, cfg):

        self.cfg = cfg
        self.device = cfg.DEVICE

        if self.device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.estimator = self.cfg.ESTIMATOR
        self.dataset_name = self.cfg.DATASET_NAME
        self.body_representation = self.cfg.BODY_REPRESENTATION

        self.vis_seq_index = self.cfg.VIS.INPUT_VIDEO_NUMBER
        self.vis_output_video_path = self.cfg.VIS.OUTPUT_VIDEO_PATH
        self.vis_input_video_path = os.path.join(self.cfg.VIS.INPUT_VIDEO_PATH,
                                                 self.dataset_name)

        self.slide_window_size = self.cfg.MODEL.SLIDE_WINDOW_SIZE
        self.slide_window_step = self.cfg.EVALUATE.SLIDE_WINDOW_STEP_SIZE

        self.ground_truth_path = eval("self.cfg.DATASET." +
                                      cfg.DATASET_NAME.upper() +
                                      ".GROUND_TRUTH_PATH")
        self.detected_path = os.path.join(
            eval("self.cfg.DATASET." + cfg.DATASET_NAME.upper() +
                 ".DETECTED_PATH"), self.estimator)

        try:
            self.ground_truth_data = np.load(os.path.join(
                self.ground_truth_path,
                self.dataset_name + "_" + "gt" + "_test.npz"),
                                             allow_pickle=True)
        except:
            raise ImportError("Ground_truth data do not exist!")

        try:
            self.detected_data = np.load(os.path.join(
                self.detected_path,
                self.dataset_name + "_" + self.estimator + "_test.npz"),
                                         allow_pickle=True)
        except:
            raise ImportError("Detected data do not exist!")

        self.device = self.cfg.DEVICE

        if self.body_representation == '3D':
            self.input_dimension = eval("self.cfg.DATASET." +
                                        cfg.DATASET_NAME.upper() +
                                        ".KEYPOINT_NUM") * 3

        elif self.body_representation == 'smpl':
            if cfg.TRAIN.USE_6D_SMPL:
                self.input_dimension = 6 * 24
            else:
                self.input_dimension = 3 * 24

        elif self.body_representation == '2D':
            self.input_dimension = eval("self.cfg.DATASET." +
                                        cfg.DATASET_NAME.upper() +
                                        ".KEYPOINT_NUM") * 2

    def visualize_3d(self, model):
        H36M_TO_J17 = [
            6, 5, 4, 1, 2, 3, 16, 15, 14, 11, 12, 13, 8, 10, 0, 7, 9
        ]
        J17_TO_J14 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]

        keypoint_number = eval("self.cfg.DATASET." +
                               self.cfg.DATASET_NAME.upper() + ".KEYPOINT_NUM")
        keypoint_root = eval("self.cfg.DATASET." +
                               self.cfg.DATASET_NAME.upper() + ".KEYPOINT_ROOT")

        data_gt = self.ground_truth_data["joints_3d"][self.vis_seq_index]
        data_pred = self.detected_data["joints_3d"][self.vis_seq_index]
                         
        if self.dataset_name == "h36m" or self.dataset_name == "pw3d":
            data_gt = data_gt.reshape(-1, 17,3)
            data_pred = data_pred.reshape(-1, 17,3)
        
        if self.dataset_name == "pw3d":
            data_gt = data_gt[:, H36M_TO_J17, :][:, J17_TO_J14, :].copy()
            data_pred = data_pred[:, H36M_TO_J17, :][:,J17_TO_J14, :].copy()

        data_gt = data_gt - data_gt[:, keypoint_root, :].mean(axis=1).reshape(-1, 1, 3)
        data_pred = data_pred - data_pred[:, keypoint_root, :].mean(axis=1).reshape(-1, 1, 3)

        data_imgname = self.ground_truth_data["imgname"][self.vis_seq_index]

        data_gt = torch.tensor(data_gt).to(self.device)
        data_pred = torch.tensor(data_pred).to(self.device)

        data_len = data_pred.shape[0]
        data_pred_window = torch.as_strided(
            data_pred, ((data_len - self.slide_window_size) // self.slide_window_step+1,
                        self.slide_window_size, keypoint_number, 3),
            (self.slide_window_step * keypoint_number * 3,
             keypoint_number * 3, 3, 1),
            storage_offset=0).reshape(-1, self.slide_window_size,
                                      self.input_dimension)

        with torch.no_grad():
            predicted_pos, denoised_pos = model(data_pred_window, self.device)

        predicted_pos = slide_window_to_sequence(predicted_pos,self.slide_window_step,self.slide_window_size).reshape(-1, keypoint_number, 3)

        data_len = predicted_pos.shape[0]
        data_pred = data_pred[:data_len, :].reshape(-1, keypoint_number, 3)
        data_gt = data_gt[:data_len, :].reshape(-1, keypoint_number, 3)

        data_imgname = data_imgname[:data_len]

        if self.dataset_name == "pw3d":
            data_gt = data_gt.reshape(-1, keypoint_number, 3)
            data_pred = data_pred.reshape(-1, keypoint_number, 3)
            predicted_pos = predicted_pos.reshape(-1, keypoint_number, 3)
            data_imgname_full = np.array([''] * data_len, dtype=object)
            for data_imgname_i in range(len(data_imgname)):
                data_imgname_full[data_imgname_i] = os.path.join(
                    self.vis_input_video_path, data_imgname[data_imgname_i])

            vis_output_video_name = "pw3d_3D_" + str(
                self.vis_seq_index) + ".mp4"

            visualize_3d_pw3d(
                data_imgname_full,
                self.vis_output_video_path,
                vis_output_video_name,
                data_pred,
                data_gt,
                predicted_pos,
                self.cfg.VIS.START,
                self.cfg.VIS.END,
            )
        elif self.dataset_name == "h36m":
            data_gt = data_gt.reshape(-1, keypoint_number, 3)
            data_pred = data_pred.reshape(-1, keypoint_number, 3)
            predicted_pos = predicted_pos.reshape(-1, keypoint_number, 3)
            data_imgname_full = np.array([''] * data_len, dtype=object)
            for data_imgname_i in range(len(data_imgname)):
                data_imgname_full[data_imgname_i] = os.path.join(
                    self.vis_input_video_path, data_imgname[data_imgname_i])

            vis_output_video_name = "h36m_3D_" + str(
                self.vis_seq_index) + ".mp4"

            visualize_3d_h36m(
                data_imgname_full,
                self.vis_output_video_path,
                vis_output_video_name,
                data_pred,
                data_gt,
                predicted_pos,
                self.cfg.VIS.START,
                self.cfg.VIS.END,
            )
        elif self.dataset_name=="aist":
            data_gt = data_gt.reshape(-1, keypoint_number, 3)
            data_pred = data_pred.reshape(-1, keypoint_number, 3)
            predicted_pos = predicted_pos.reshape(-1, keypoint_number, 3)
            data_imgname_full = np.array([''] * data_len, dtype=object)
            for data_imgname_i in range(len(data_imgname)):
                data_imgname_full[data_imgname_i] = os.path.join(
                    self.vis_input_video_path, data_imgname[data_imgname_i])

            vis_output_video_name = "aist_3D_" + str(
                self.vis_seq_index) + ".mp4"

            visualize_3d_aist(
                data_imgname_full,
                self.vis_output_video_path,
                vis_output_video_name,
                data_pred,
                data_gt,
                predicted_pos,
                self.cfg.VIS.START,
                self.cfg.VIS.END,
            )
        else:
            print("Not Implemented!")

    def visualize_smpl(self, model):

        data_gt = self.ground_truth_data["pose"][self.vis_seq_index]
        data_pred = self.detected_data["pose"][self.vis_seq_index]

        if self.cfg.TRAIN.USE_6D_SMPL:
            data_pred = numpy_axis_to_rot6D(data_pred.reshape(-1, 3)).reshape(
                -1, self.input_dimension)

        data_imgname = self.ground_truth_data["imgname"][self.vis_seq_index]

        data_gt = torch.tensor(data_gt).to(self.device)
        data_pred = torch.tensor(data_pred).to(self.device)

        data_len = data_pred.shape[0]
        data_pred_window = torch.as_strided(
            data_pred, ((data_len - self.slide_window_size) // self.slide_window_step+1,
                        self.slide_window_size, self.input_dimension),
            (self.slide_window_step * self.input_dimension,
             self.input_dimension, 1),
            storage_offset=0).reshape(-1, self.slide_window_size,
                                      self.input_dimension)

        with torch.no_grad():
            predicted_pos, denoised_pos = model(data_pred_window, self.device)

        predicted_pos = slide_window_to_sequence(predicted_pos,self.slide_window_step,self.slide_window_size).reshape(-1, self.input_dimension)

        data_len = predicted_pos.shape[0]
        data_pred = data_pred[:data_len, :]
        data_gt = data_gt[:data_len, :]

        data_imgname = data_imgname[:data_len]

        if self.cfg.TRAIN.USE_6D_SMPL:
            data_pred = rot6D_to_axis(data_pred.reshape(-1, 6)).reshape(
                -1, 24 * 3)
            predicted_pos = rot6D_to_axis(predicted_pos.reshape(-1,
                                                                6)).reshape(
                                                                    -1, 24 * 3)

        data_gt = np.array(data_gt.reshape(-1, 24 * 3).cpu())
        data_pred = np.array(data_pred.reshape(-1, 24 * 3).cpu())
        predicted_pos = np.array(predicted_pos.reshape(-1, 24 * 3).cpu())

        smpl_neural = SMPL(model_path=self.cfg.SMPL_MODEL_DIR,
                           create_transl=False)

        if self.dataset_name == "pw3d":
            data_imgname_full = np.array([''] * data_len, dtype=object)
            vis_output_video_name = "pw3d_smpl_" + str(
                self.vis_seq_index) + ".mp4"
            for data_imgname_i in range(len(data_imgname)):
                data_imgname_full[data_imgname_i] = os.path.join(
                    self.vis_input_video_path, data_imgname[data_imgname_i])

            visualize_smpl_pw3d(
                data_imgname_full,
                self.vis_output_video_path,
                vis_output_video_name,
                smpl_neural,
                data_pred,
                data_gt,
                predicted_pos,
                self.cfg.VIS.START,
                self.cfg.VIS.END,
            )
        if self.dataset_name == "aist":
            data_imgname_full = np.array([''] * data_len, dtype=object)
            vis_output_video_name = "aist_smpl_" + str(
                self.vis_seq_index) + ".mp4"
            for data_imgname_i in range(len(data_imgname)):
                data_imgname_full[data_imgname_i] = os.path.join(
                    self.vis_input_video_path, data_imgname[data_imgname_i])

            visualize_smpl_aist(
                data_imgname_full,
                self.vis_output_video_path,
                vis_output_video_name,
                smpl_neural,
                data_pred,
                data_gt,
                predicted_pos,
                self.cfg.VIS.START,
                self.cfg.VIS.END,
            )

    def visualize_2d(self, model):
        keypoint_number = eval("self.cfg.DATASET." +
                               self.cfg.DATASET_NAME.upper() + ".KEYPOINT_NUM")

        data_gt = self.ground_truth_data["joints_2d"][self.vis_seq_index]
        data_pred = self.detected_data["joints_2d"][self.vis_seq_index]

        data_imgname = self.ground_truth_data["imgname"][self.vis_seq_index]
        data_bbox = self.ground_truth_data["bbox"][self.vis_seq_index]
        data_imageshape=self.ground_truth_data["imgshape"][self.vis_seq_index]

        data_gt = torch.tensor(data_gt).to(self.device)
        data_pred_norm=torch.tensor(data_pred.reshape(-1,2)/data_imageshape[:2][::-1]).to(self.device).reshape_as(data_gt)
        data_pred = torch.tensor(data_pred).to(self.device)

        data_len = data_pred.shape[0]
        data_pred_window = torch.as_strided(
            data_pred_norm, ((data_len - self.slide_window_size) // self.slide_window_step+1,
                        self.slide_window_size, keypoint_number, 2),
            (self.slide_window_step * keypoint_number * 2,
             keypoint_number * 2, 2, 1),
            storage_offset=0).reshape(-1, self.slide_window_size,
                                      self.input_dimension)

        with torch.no_grad():
            predicted_pos, denoised_pos = model(data_pred_window, self.device)

        predicted_pos = slide_window_to_sequence(predicted_pos,self.slide_window_step,self.slide_window_size).reshape(-1, keypoint_number, 2)

        data_len = predicted_pos.shape[0]
        data_pred = data_pred[:data_len, :].reshape(-1, keypoint_number, 2)
        data_gt = data_gt[:data_len, :].reshape(-1, keypoint_number, 2)

        data_imgname = data_imgname[:data_len]

        data_gt = np.array(data_gt.reshape(-1, keypoint_number, 2).cpu())
        data_pred = np.array(data_pred.reshape(-1, keypoint_number, 2).cpu())
        predicted_pos = np.array(
            predicted_pos.reshape(-1, keypoint_number, 2).cpu())*data_imageshape[:2][::-1]

        if self.dataset_name == "jhmdb":
            data_imgname_full = np.array([''] * data_len, dtype=object)
            for data_imgname_i in range(len(data_imgname)):
                data_imgname_split = data_imgname[data_imgname_i].split('/')
                data_imgname_full[data_imgname_i] = os.path.join(
                    self.vis_input_video_path, data_imgname_split[0],
                    data_imgname_split[1],
                    f"{int(data_imgname_split[2])+1:05d}.png")
            vis_output_video_name = "jhmdb_2D_" + str(
                self.vis_seq_index) + ".mp4"
            visualize_2d_jhmdb(
                data_imgname_full,
                self.vis_output_video_path,
                vis_output_video_name,
                predicted_pos,
                data_pred,
                data_gt,
                data_bbox,
                self.cfg.VIS.START,
                self.cfg.VIS.END,
            )
        else:
            print("Not Implemented!")

    def visualize(self, model):
        model.eval()
        if self.cfg.BODY_REPRESENTATION == "3D":
            self.visualize_3d(model)

        elif self.cfg.BODY_REPRESENTATION == "smpl":
            self.visualize_smpl(model)

        elif self.cfg.BODY_REPRESENTATION == "2D":
            self.visualize_2d(model)
