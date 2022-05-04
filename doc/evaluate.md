# Evaluation Detail

Training configs are explained in [lib/core/config.py](../lib/core/config.py). Different configs for different datasets and estimators are in [config](../configs/) folder.

The corresponding checkpoints can be download here and they are supposed to be put under ```data/checkpoints```:

| Dataset | Pose Estimator | 3D Keypoints | 2D Keypoints | SMPL |
|  ----  | ----  | ----  | ----  | ----  |
| [Sub-JHMDB](http://jhmdb.is.tue.mpg.de/)  | [SimplePose](https://openaccess.thecvf.com/content_ECCV_2018/html/Bin_Xiao_Simple_Baselines_for_ECCV_2018_paper.html) |   | [Baidu Netdisk](https://pan.baidu.com/s/1W_9xEyJ9Y7zlBOt5fYpEWQ?pwd=rehu) / [Google Drive](https://drive.google.com/drive/folders/1Wd4MxpxLmqoTMB8AlnnMY4Vb641dp2Tw?usp=sharing) |   |
| [3DPW](https://virtualhumans.mpi-inf.mpg.de/3DPW/)  | [EFT](https://github.com/facebookresearch/eft) | [Baidu Netdisk](https://pan.baidu.com/s/1d5Ib-IgWVPRbjUOf9LFXug?pwd=w3v2) / [Google Drive](https://drive.google.com/drive/folders/17xO_X213hcNEEtJbJlz8qE2aCB3-gncH?usp=sharing) |   |  [Baidu Netdisk](https://pan.baidu.com/s/1SP9EPwd_S0MPiyTfWGLgUg?pwd=8lfn) / [Google Drive](https://drive.google.com/drive/folders/1P_LObi8Tr09lw8149Pqe4Ks2SOK-RvYN?usp=sharing) |
| [3DPW](https://virtualhumans.mpi-inf.mpg.de/3DPW/)  | [PARE](https://pare.is.tue.mpg.de/) | [Baidu Netdisk](https://pan.baidu.com/s/1gePXz93tT74GQbfmSStg4Q?pwd=ug8m) / [Google Drive](https://drive.google.com/drive/folders/19E-5lfPHRUelIc2vgdu-M_CEatdFNul_?usp=sharing) |   | [Baidu Netdisk](https://pan.baidu.com/s/1Leo2O1FHoumk0lMaX9AFhQ?pwd=7504) / [Google Drive](https://drive.google.com/drive/folders/1m7IeojeAN9_WBTCwv8921RgOX1SPn7P4?usp=sharing) |
| [3DPW](https://virtualhumans.mpi-inf.mpg.de/3DPW/)  | [SPIN](https://github.com/nkolot/SPIN) | [Baidu Netdisk](https://pan.baidu.com/s/1Kj70V107nGBH7142onXODQ?pwd=9p4o) / [Google Drive](https://drive.google.com/drive/folders/1lj93zsJj3_InTFGWpyNNZ_R7gRQSZE4P?usp=sharing) |   | [Baidu Netdisk](https://pan.baidu.com/s/1obQaCp6yjdkMQr2FRF3Y2A?pwd=b8ur) / [Google Drive](https://drive.google.com/drive/folders/1j7pYCOvvzBBcpu7G_S5-GOXenSXaDeZl?usp=sharing) |
| [Human3.6M](http://vision.imar.ro/human3.6m/description.php)  | [FCN](https://github.com/una-dinosauria/3d-pose-baseline) | [Baidu Netdisk](https://pan.baidu.com/s/1B_yLjyzNVNlE4fQOHuLTFQ?pwd=gdek) / [Google Drive](https://drive.google.com/drive/folders/1LblRGrXeVnW3jDwgYD9hj-ladhnumCrW?usp=sharing) |   |   |
| [AIST++](https://google.github.io/aistplusplus_dataset/factsfigures.html)  | [SPIN](https://github.com/nkolot/SPIN) | [Baidu Netdisk](https://pan.baidu.com/s/1X2KvDirfq5lIE9yrlbIbqg?pwd=5jpi) / [Google Drive](https://drive.google.com/drive/folders/17JNAyJqHx577oP4fWFUQHQIjIjFUuf6v?usp=sharing) |   | [Baidu Netdisk](https://pan.baidu.com/s/1EwiR3AyMP8tnSYgU1VY1Tg?pwd=7p4f) / [Google Drive](https://drive.google.com/drive/folders/1X8N1XU2IN3DMSEE5u36Ca8nkuKEul5hj?usp=sharing) |

We also provide checkpoints with different intervals and slide window Q for different datasets and backbones. ```checkpoint_i3_q33.pth.tar``` means checkpoint trained with ```interval=3``` and```slide window Q=33```. You can find the evaluation results of different intervals below.

## Evaluation Commands

You can directly evaluate the model in different datasets and estimator settings using following commands

### 2D
Sub-JHMDB Simplepose
```shell script
python eval.py --cfg configs/config_jhmdb_simplepose_2D.yaml --dataset_name jhmdb --estimator simplepose --body_representation 2D --sample_interval 10

```

### 3D
3DPW SPIN
```shell script
python eval.py --cfg configs/config_pw3d_spin_3D.yaml --dataset_name pw3d --estimator spin --body_representation 3D --sample_interval 10

```
3DPW EFT
```shell script
python eval.py --cfg configs/config_pw3d_eft_3D.yaml --dataset_name pw3d --estimator eft --body_representation 3D --sample_interval 10

```
3DPW PARE
```shell script
python eval.py --cfg configs/config_pw3d_pare_3D.yaml --dataset_name pw3d --estimator pare --body_representation 3D --sample_interval 10

```
AIST++ SPIN
```shell script
python eval.py --cfg configs/config_aist_spin_3D.yaml --dataset_name aist --estimator spin --body_representation 3D --sample_interval 10

```
Human3.6M FCN
```shell script
python eval.py --cfg configs/config_h36m_fcn_3D.yaml --dataset_name h36m --estimator fcn --body_representation 3D --sample_interval 10

```

### SMPL
3DPW SPIN
```shell script
python eval.py --cfg configs/config_pw3d_spin_smpl.yaml --dataset_name pw3d --estimator spin --body_representation smpl --sample_interval 10

```
3DPW EFT
```shell script
python eval.py --cfg configs/config_pw3d_eft_smpl.yaml --dataset_name pw3d --estimator eft --body_representation smpl --sample_interval 10

```
3DPW PARE
```shell script
python eval.py --cfg configs/config_pw3d_pare_smpl.yaml --dataset_name pw3d --estimator pare --body_representation smpl --sample_interval 10

```
AIST++ SPIN
```shell script
python eval.py --cfg configs/config_aist_spin_smpl.yaml --dataset_name aist --estimator spin --body_representation smpl --sample_interval 10

```

## Useful configs

- Set ```cfg.EVALUATE.INTERP = True``` to see the performance of the linear interpolatio.

- Set ```cfg.EVALUATE.RELATIVE_IMPROVEMENT = True``` to see the relative improvemrnt of DeciWatch, which is calculated as:

    $$
    relative \ improvement =  mean(DeciWatch \ output \ error) - mean(selected \ frames \ input \ error)
    $$


- Set ```cfg.EVALUATE.DENOISE = True``` to see the performance of DeciWatch DenoiseNet.

## Different interval results

We are stilling working on the blanked results.

### Sub-JHMDB Simplepose 2D

| Interval/Q | 1/10 | 2/5 | 3/3 | 4/2 | 5/2 | 6/1 | 7/1 | 8/1 | 9/1 | 10/1 | 11/1 | 12/1 | 13/1 | 14/1 | 15/1 | 16/1 | 17/1 | 18/1 | 19/1 | 20/1 |
| ---------- | ----- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ----- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- |
| MPJPE(IN/OUT)     |       |      |      |      |      |      |      |      |      |       |      |      |      |      |      |      |      |      |      |      |
|  ACC(IN/OUT)   |       |      |      |      |      |      |      |      |      |       |      |      |      |      |      |      |      |      |      |      |


### PW3D SPIN 3D

| Interval/Q | 1/100 | 2/50 | 3/33 | 4/25 | 5/20 | 6/16 | 7/14 | 8/12 | 9/11 | 10/10 | 11/9 | 12/8 | 13/7 | 14/7 | 15/6 | 16/6 | 17/5 | 18/5 | 19/5 | 20/5 |
| ---------- | ----- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ----- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- |
| MPJPE(IN/OUT)     |       |      |      |      |      |      |      |      |      |       |      |      |      |      |      |      |      |      |      |      |
|  ACC(IN/OUT)   |       |      |      |      |      |      |      |      |      |       |      |      |      |      |      |      |      |      |      |      |


### PW3D EFT 3D

| Interval/Q | 1/100 | 2/50 | 3/33 | 4/25 | 5/20 | 6/16 | 7/14 | 8/12 | 9/11 | 10/10 | 11/9 | 12/8 | 13/7 | 14/7 | 15/6 | 16/6 | 17/5 | 18/5 | 19/5 | 20/5 |
| ---------- | ----- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ----- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- |
| MPJPE(IN/OUT)     |       |      |      |      |      |      |      |      |      |       |      |      |      |      |      |      |      |      |      |      |
|  ACC(IN/OUT)   |       |      |      |      |      |      |      |      |      |       |      |      |      |      |      |      |      |      |      |      |



### PW3D PARE 3D

| Interval/Q | 1/100 | 2/50 | 3/33 | 4/25 | 5/20 | 6/16 | 7/14 | 8/12 | 9/11 | 10/10 | 11/9 | 12/8 | 13/7 | 14/7 | 15/6 | 16/6 | 17/5 | 18/5 | 19/5 | 20/5 |
| ---------- | ----- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ----- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- |
| MPJPE(IN/OUT)     |       |      |      |      |      |      |      |      |      |       |      |      |      |      |      |      |      |      |      |      |
|  ACC(IN/OUT)   |       |      |      |      |      |      |      |      |      |       |      |      |      |      |      |      |      |      |      |      |



### H36M FCN 3D

| Interval/Q | 1/100 | 2/50 | 3/33 | 4/25 | 5/20 | 6/16 | 7/14 | 8/12 | 9/11 | 10/10 | 11/9 | 12/8 | 13/7 | 14/7 | 15/6 | 16/6 | 17/5 | 18/5 | 19/5 | 20/5 |
| ---------- | ----- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ----- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- |
| MPJPE(IN/OUT)     |       |      |      |      |      |      |      |      |      |       |      |      |      |      |      |      |      |      |      |      |
|  ACC(IN/OUT)   |       |      |      |      |      |      |      |      |      |       |      |      |      |      |      |      |      |      |      |      |



### AIST SPIN 3D

| Interval/Q | 1/100 | 2/50 | 3/33 | 4/25 | 5/20 | 6/16 | 7/14 | 8/12 | 9/11 | 10/10 | 11/9 | 12/8 | 13/7 | 14/7 | 15/6 | 16/6 | 17/5 | 18/5 | 19/5 | 20/5 |
| ---------- | ----- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ----- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- |
| MPJPE(IN/OUT)     |       |      |      |      |      |      |      |      |      |       |      |      |      |      |      |      |      |      |      |      |
|  ACC(IN/OUT)   |       |      |      |      |      |      |      |      |      |       |      |      |      |      |      |      |      |      |      |      |



### PW3D SPIN SMPL

| Interval/Q | 1/100 | 2/50 | 3/33 | 4/25 | 5/20 | 6/16 | 7/14 | 8/12 | 9/11 | 10/10 | 11/9 | 12/8 | 13/7 | 14/7 | 15/6 | 16/6 | 17/5 | 18/5 | 19/5 | 20/5 |
| ---------- | ----- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ----- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- |
| MPJPE(IN/OUT)     |       |      |      |      |      |      |      |      |      |       |      |      |      |      |      |      |      |      |      |      |
|  ACC(IN/OUT)   |       |      |      |      |      |      |      |      |      |       |      |      |      |      |      |      |      |      |      |      |



### PW3D EFT SMPL

| Interval/Q | 1/100 | 2/50 | 3/33 | 4/25 | 5/20 | 6/16 | 7/14 | 8/12 | 9/11 | 10/10 | 11/9 | 12/8 | 13/7 | 14/7 | 15/6 | 16/6 | 17/5 | 18/5 | 19/5 | 20/5 |
| ---------- | ----- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ----- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- |
| MPJPE(IN/OUT)     |       |      |      |      |      |      |      |      |      |       |      |      |      |      |      |      |      |      |      |      |
|  ACC(IN/OUT)   |       |      |      |      |      |      |      |      |      |       |      |      |      |      |      |      |      |      |      |      |



### PW3D PARE SMPL

| Interval/Q | 1/100 | 2/50 | 3/33 | 4/25 | 5/20 | 6/16 | 7/14 | 8/12 | 9/11 | 10/10 | 11/9 | 12/8 | 13/7 | 14/7 | 15/6 | 16/6 | 17/5 | 18/5 | 19/5 | 20/5 |
| ---------- | ----- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ----- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- |
| MPJPE(IN/OUT)     |       |      |      |      |      |      |      |      |      |       |      |      |      |      |      |      |      |      |      |      |
|  ACC(IN/OUT)   |       |      |      |      |      |      |      |      |      |       |      |      |      |      |      |      |      |      |      |      |



### AIST SPIN SMPL

| Interval/Q | 1/100 | 2/50 | 3/33 | 4/25 | 5/20 | 6/16 | 7/14 | 8/12 | 9/11 | 10/10 | 11/9 | 12/8 | 13/7 | 14/7 | 15/6 | 16/6 | 17/5 | 18/5 | 19/5 | 20/5 |
| ---------- | ----- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ----- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- |
| MPJPE(IN/OUT)     |       |      |      |      |      |      |      |      |      |       |      |      |      |      |      |      |      |      |      |      |
|  ACC(IN/OUT)   |       |      |      |      |      |      |      |      |      |       |      |      |      |      |      |      |      |      |      |      |




