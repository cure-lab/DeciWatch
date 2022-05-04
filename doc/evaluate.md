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
| PCK0.2 :arrow_up_small:    |       |      |      |      |      |      |      |      |      |       |      |      |      |      |      |      |      |      |      |      |
|  PCK0.1 :arrow_up_small:  |       |      |      |      |      |      |      |      |      |       |      |      |      |      |      |      |      |      |      |      |
|  PCK0.05 :arrow_up_small:  |       |      |      |      |      |      |      |      |      |       |      |      |      |      |      |      |      |      |      |      |


### PW3D SPIN 3D

| Interval/Q | 1/100 | 2/50 | 3/33 | 4/25 | 5/20 | 6/16 | 7/14 | 8/12 | 9/11 | 10/10 | 11/9 | 12/8 | 13/7 | 14/7 | 15/6 | 16/6 | 17/5 | 18/5 | 19/5 | 20/5 |
| ---------- | ----- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ----- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- |
| MPJPE(IN/OUT) :arrow_down_small:    |   96.85/93.82    |   96.85/93.23   |  96.87/92.86    |  96.87/92.94    |  96.89/92.20    |   96.89/92.72   |  96.92/92.58    |  96.91/93.21    |   96.92/93.42   |  96.92/93.37     |  96.96/94.20    |   96.97/95.00   |  96.96/95.29    |   96.98/96.59   |  97.00/96.66    |  96.99/97.36    |   96.99/97.59   |  97.05/98.84    |  97.00/99.19    |   97.00/100.18   |
|  ACC(IN/OUT) :arrow_down_small:  |  34.62/33.52     | 34.62/14.47     |  34.64/10.51    |  34.64/8.94    |  34.65/8.15    |   34.66/7.73   |   34.67/7.45   |  34.68/7.27    | 34.67/7.17     |   34.68/7.05    |   34.70/7.00   |  34.71/6.93    |   34.70/6.90   |  34.73/6.85    |  34.74/6.87    |  34.74/6.81    |  34.73/6.85    |   34.78/6.82   |  34.75/6.77    |  34.75/6.76    |


### PW3D EFT 3D

| Interval/Q | 1/100 | 2/50 | 3/33 | 4/25 | 5/20 | 6/16 | 7/14 | 8/12 | 9/11 | 10/10 | 11/9 | 12/8 | 13/7 | 14/7 | 15/6 | 16/6 | 17/5 | 18/5 | 19/5 | 20/5 |
| ---------- | ----- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ----- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- |
| MPJPE(IN/OUT) :arrow_down_small:    |   90.32/88.02    |  90.32/87.44    |  90.33/87.18    |   90.32/87.16    |  90.33/87.24    |  90.33/87.31    |  90.33/87.58    |  90.32/88.32    |  90.32/88.38    |   90.34/89.02    |  90.31/89.83    |  90.33/90.25    |  90.34/91.48    |  90.35/92.30    |     90.36/92.28   |   90.31/93.05   |   90.34/94.04   |  90.37/94.59    |  90.35/95.43   | 90.34/96.10  |
|  ACC(IN/OUT) :arrow_down_small:  |  32.78/32.69     |  32.78/14.09    |  32.79/10.15    |   32.78/8.67    |  32.80/7.96    |  32.81/7.53    |  32.80/7.27    |  32.82/7.06    |  32.80/6.94    |  32.83/6.84     |  32.83/6.79    |   32.84/6.70   | 32.85/6.65     |    32.86/6.60   |  32.88/6.57    |   32.88/6.51   |  32.84/6.51    |  32.90/6.47    |  32.88/6.44    | 32.88/6.42 |



### PW3D PARE 3D

| Interval/Q | 1/100 | 2/50 | 3/33 | 4/25 | 5/20 | 6/16 | 7/14 | 8/12 | 9/11 | 10/10 | 11/9 | 12/8 | 13/7 | 14/7 | 15/6 | 16/6 | 17/5 | 18/5 | 19/5 | 20/5 |
| ---------- | ----- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ----- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- |
| MPJPE(IN/OUT) :arrow_down_small:    |   78.91/75.73    |  78.91/75.24    |  78.92/75.02    |  78.93/74.91    |  78.94/75.14    |   78.95/75.22   |   78.97/75.41   |  78.96/76.02    |   78.97/76.39   |   78.98/77.16    |   78.99/77.74   |  79.02/78.38    |  78.99/79.43    |   79.02/80.25   |   79.03/80.70   |   79.02/81.66   |   79.01/82.53   |  79.08/83.52    |   79.06/84.40   |  79.03/85.25    |
|  ACC(IN/OUT) :arrow_down_small:  |  25.69/25.18     |   25.70/11.95   |   25.71/9.23   |  25.71/8.22    |  25.72/7.72    |  25.73/7.40    |  25.74/7.21    |  25.74/7.07    |   25.74/6.98   |   25.75/6.90    |  25.77/6.85    |  25.77/6.80    |   25.76/6.76   |   25.78/6.71   |  25.78/6.70    |  25.79/6.66    |  25.77/6.67    |  25.81/6.63    |   25.81/6.61   |   25.80/6.59   |



### H36M FCN 3D

| Interval/Q | 1/100 | 2/50 | 3/33 | 4/25 | 5/20 | 6/16 | 7/14 | 8/12 | 9/11 | 10/10 | 11/9 | 12/8 | 13/7 | 14/7 | 15/6 | 16/6 | 17/5 | 18/5 | 19/5 | 20/5 |
| ---------- | ----- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ----- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- |
| MPJPE(IN/OUT)  :arrow_down_small:   |       |      |      |      |      |      |      |      |      |       |      |      |      |      |      |      |      |      |      |      |
|  ACC(IN/OUT)  :arrow_down_small: |       |      |      |      |      |      |      |      |      |       |      |      |      |      |      |      |      |      |      |      |



### AIST SPIN 3D

| Interval/Q | 1/100 | 2/50 | 3/33 | 4/25 | 5/20 | 6/16 | 7/14 | 8/12 | 9/11 | 10/10 | 11/9 | 12/8 | 13/7 | 14/7 | 15/6 | 16/6 | 17/5 | 18/5 | 19/5 | 20/5 |
| ---------- | ----- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ----- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- |
| MPJPE(IN/OUT) :arrow_down_small:    |       |      |      |      |      |      |      |      |      |       |      |      |      |      |      |      |      |      |      |      |
|  ACC(IN/OUT) :arrow_down_small:  |       |      |      |      |      |      |      |      |      |       |      |      |      |      |      |      |      |      |      |      |



### PW3D SPIN SMPL

| Interval/Q | 1/100 | 2/50 | 3/33 | 4/25 | 5/20 | 6/16 | 7/14 | 8/12 | 9/11 | 10/10 | 11/9 | 12/8 | 13/7 | 14/7 | 15/6 | 16/6 | 17/5 | 18/5 | 19/5 | 20/5 |
| ---------- | ----- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ----- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- |
| MPJPE(IN/OUT) :arrow_down_small:    |   100.13/96.80    |  100.13/96.36    |   100.03/95.95   |    100.13/96.18  |   100.13/96.10   |   100.24/96.38   |  100.29/97.02    |  100.24/97.47    |   100.03/97.42   |   100.13/97.76   |  100.03/97.93    |  100.24/99.34    |   100.24/99.76   |  100.29/100.49    |  100.30/101.02    |   100.24/101.78   |   100.12/102.13   |  100.30/103.19    |    100.11/102.46  |  100.13/104.21    |
|  ACC(IN/OUT) :arrow_down_small:  |   35.53/36.20    |   35.53/17.07   |   35.46/12.50   |  35.53/10.91    |  35.53/9.82    |  35.67/9.36    |   35.67/9.02    |  35.67/8.83    |  35.46/8.50    |  35.53/8.38     |  35.46/8.34    |  35.67/8.23    |  35.49/8.16    |   35.67/8.15   |  35.60/8.08    |  35.67/8.07    |   35.52/8.02   |  35.60/8.01    |  35.51/7.88    |  35.53/7.81    |



### PW3D EFT SMPL

| Interval/Q | 1/100 | 2/50 | 3/33 | 4/25 | 5/20 | 6/16 | 7/14 | 8/12 | 9/11 | 10/10 | 11/9 | 12/8 | 13/7 | 14/7 | 15/6 | 16/6 | 17/5 | 18/5 | 19/5 | 20/5 |
| ---------- | ----- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ----- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- |
| MPJPE(IN/OUT)  :arrow_down_small:   |       |      |      |      |      |      |      |      |      |       |      |      |      |      |      |      |      |      |      |      |
|  ACC(IN/OUT) :arrow_down_small:  |       |      |      |      |      |      |      |      |      |       |      |      |      |      |      |      |      |      |      |      |



### PW3D PARE SMPL

| Interval/Q | 1/100 | 2/50 | 3/33 | 4/25 | 5/20 | 6/16 | 7/14 | 8/12 | 9/11 | 10/10 | 11/9 | 12/8 | 13/7 | 14/7 | 15/6 | 16/6 | 17/5 | 18/5 | 19/5 | 20/5 |
| ---------- | ----- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ----- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- |
| MPJPE(IN/OUT)  :arrow_down_small:   |       |      |      |      |      |      |      |      |      |       |      |      |      |      |      |      |      |      |      |      |
|  ACC(IN/OUT) :arrow_down_small:  |       |      |      |      |      |      |      |      |      |       |      |      |      |      |      |      |      |      |      |      |



### AIST SPIN SMPL

| Interval/Q | 1/100 | 2/50 | 3/33 | 4/25 | 5/20 | 6/16 | 7/14 | 8/12 | 9/11 | 10/10 | 11/9 | 12/8 | 13/7 | 14/7 | 15/6 | 16/6 | 17/5 | 18/5 | 19/5 | 20/5 |
| ---------- | ----- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ----- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- |
| MPJPE(IN/OUT)  :arrow_down_small:   |       |      |      |      |      |      |      |      |      |       |      |      |      |      |      |      |      |      |      |      |
|  ACC(IN/OUT)  :arrow_down_small: |       |      |      |      |      |      |      |      |      |       |      |      |      |      |      |      |      |      |      |      |




