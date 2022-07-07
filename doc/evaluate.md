# Evaluation Detail

Training configs are explained in [lib/core/config.py](../lib/core/config.py). Different configs for different datasets and estimators are in [config](../configs/) folder.

The corresponding checkpoints can be download here and they are supposed to be put under ```data/checkpoints```:

| Dataset | Pose Estimator | 3D Pose | 2D Pose | SMPL |
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

### 2D Pose
Sub-JHMDB Simplepose
```shell script
python eval.py --cfg configs/config_jhmdb_simplepose_2D.yaml --dataset_name jhmdb --estimator simplepose --body_representation 2D --sample_interval 10

```

### 3D Pose
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

### Sub-JHMDB Simplepose 2D Pose

| Interval/Q | 1/10 | 2/5 | 3/3 | 4/2 | 5/2 | 6/1 | 7/1 | 8/1 | 9/1 | 10/1 | 11/1 | 12/1 | 13/1 | 14/1 | 15/1 | 16/1 | 17/1 | 18/1 | 19/1 | 20/1 |
| ---------- | ----- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ----- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- |
| PCK0.2(IN/OUT) :arrow_up:    | 93.76%/98.96% |   93.77%/98.79%   | 93.81%/98.98% | 93.80%/98.92% | 93.92%/99.11% | 93.86%/98.92% | 93.83%/98.97% | 93.75%/98.85% | 93.72%/98.86%  | 93.94%/98.75% | 93.92%/98.77% | 93.92%/98.86% | 93.89%/98.66% |   93.98%/98.87%   | 93.94%/98.73% | 94.04%/98.63% |   94.03%/98.20%   | 94.09%/98.02% |   94.11%/97.87%   | 92.38%/97.50% |
|  PCK0.1(IN/OUT) :arrow_up:  | 80.94%/95.51% | 80.99%/95.17% |   81.05%/95.40%   | 81.09%/94.90% | 81.25%/95.43% | 81.27%/94.44% |   81.14%/94.49%   | 81.27%/94.13% |   81.18%/94.21%   | 81.61%/94.05% | 81.44%/94.06% | 81.39%/93.71% | 81.27%/93.52% |   81.51%/94.14%   | 81.53%/94.12% |   81.55%/93.42%   | 81.54%/92.26% | 81.75%/92.24% | 81.81%/91.68%  | 82.79%/91.76% |
|  PCK0.05(IN/OUT) :arrow_up:  | 56.56%/85.32% | 56.63%/83.80% |   56.64%/83.42%   | 56.69%/82.57% | 56.88%/82.66% | 56.85%/80.27% | 56.78%/80.12% |   56.92%/79.36%   |   56.73%/79.32%   | 57.30%/79.44% | 57.16%/78.98% |   57.01%/77.65%   | 56.94%/77.15% | 57.12%/78.32%  | 57.25%/78.36% | 57.25%/76.98% | 57.35%/75.16% |   57.50%/74.51%   | 57.65%/74.21% | 58.95%/73.02% |


### PW3D SPIN 3D Pose

| Interval/Q | 1/100 | 2/50 | 3/33 | 4/25 | 5/20 | 6/16 | 7/14 | 8/12 | 9/11 | 10/10 | 11/9 | 12/8 | 13/7 | 14/7 | 15/6 | 16/6 | 17/5 | 18/5 | 19/5 | 20/5 |
| ---------- | ----- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ----- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- |
| MPJPE(IN/OUT) :arrow_down:    |   96.85/93.82    |   96.85/93.23   |  96.87/92.86    |  96.87/92.94    |  96.89/92.20    |   96.89/92.72   |  96.92/92.58    |  96.91/93.21    |   96.92/93.42   | 96.92/93.34 |  96.96/94.20    |   96.97/95.00   |  96.96/95.29    |   96.98/96.59   |  97.00/96.66    |  96.99/97.36    |   96.99/97.59   |  97.05/98.84    |  97.00/99.19    |   97.00/100.18   |
|  ACC(IN/OUT) :arrow_down:  |  34.62/33.52     | 34.62/14.47     |  34.64/10.51    |  34.64/8.94    |  34.65/8.15    |   34.66/7.73   |   34.67/7.45   |  34.68/7.27    | 34.67/7.17     | 34.68/7.06 |   34.70/7.00   |  34.71/6.93    |   34.70/6.90   |  34.73/6.85    |  34.74/6.87    |  34.74/6.81    |  34.73/6.85    |   34.78/6.82   |  34.75/6.77    |  34.75/6.76    |


### PW3D EFT 3D Pose

| Interval/Q | 1/100 | 2/50 | 3/33 | 4/25 | 5/20 | 6/16 | 7/14 | 8/12 | 9/11 | 10/10 | 11/9 | 12/8 | 13/7 | 14/7 | 15/6 | 16/6 | 17/5 | 18/5 | 19/5 | 20/5 |
| ---------- | ----- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ----- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- |
| MPJPE(IN/OUT) :arrow_down:    |   90.32/88.02    |  90.32/87.44    |  90.33/87.18    |   90.32/87.16    |  90.33/87.24    |  90.33/87.31    |  90.33/87.58    |  90.32/88.32    |  90.32/88.38    | 90.34/89.02 |  90.31/89.83    |  90.33/90.25    |  90.34/91.48    |  90.35/92.30    |     90.36/92.28   |   90.31/93.05   |   90.34/94.04   |  90.37/94.59    |  90.35/95.43   | 90.34/96.10  |
|  ACC(IN/OUT) :arrow_down:  |  32.78/32.69     |  32.78/14.09    |  32.79/10.15    |   32.78/8.67    |  32.80/7.96    |  32.81/7.53    |  32.80/7.27    |  32.82/7.06    |  32.80/6.94    |  32.83/6.84     |  32.83/6.79    |   32.84/6.70   | 32.85/6.65     |    32.86/6.60   |  32.88/6.57    |   32.88/6.51   |  32.84/6.51    |  32.90/6.47    |  32.88/6.44    | 32.88/6.42 |



### PW3D PARE 3D Pose

| Interval/Q | 1/100 | 2/50 | 3/33 | 4/25 | 5/20 | 6/16 | 7/14 | 8/12 | 9/11 | 10/10 | 11/9 | 12/8 | 13/7 | 14/7 | 15/6 | 16/6 | 17/5 | 18/5 | 19/5 | 20/5 |
| ---------- | ----- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ----- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- |
| MPJPE(IN/OUT) :arrow_down:    |   78.91/75.73    |  78.91/75.24    |  78.92/75.02    |  78.93/74.91    |  78.94/75.14    |   78.95/75.22   |   78.97/75.41   |  78.96/76.02    |   78.97/76.39   |   78.98/77.16    |   78.99/77.74   |  79.02/78.38    |  78.99/79.43    |   79.02/80.25   |   79.03/80.70   |   79.02/81.66   |   79.01/82.53   |  79.08/83.52    |   79.06/84.40   |  79.03/85.25    |
|  ACC(IN/OUT) :arrow_down:  |  25.69/25.18     |   25.70/11.95   |   25.71/9.23   |  25.71/8.22    |  25.72/7.72    |  25.73/7.40    |  25.74/7.21    |  25.74/7.07    |   25.74/6.98   |   25.75/6.90    |  25.77/6.85    |  25.77/6.80    |   25.76/6.76   |   25.78/6.71   |  25.78/6.70    |  25.79/6.66    |  25.77/6.67    |  25.81/6.63    |   25.81/6.61   |   25.80/6.59   |



### H36M FCN 3D Pose

| Interval/Q | 1/100 | 2/50 | 3/33 | 4/25 | 5/20 | 6/16 | 7/14 | 8/12 | 9/11 | 10/10 | 11/9 | 12/8 | 13/7 | 14/7 | 15/6 | 16/6 | 17/5 | 18/5 | 19/5 | 20/5 |
| ---------- | ----- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ----- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- |
| MPJPE(IN/OUT)  :arrow_down:   |   54.55/53.25    |  54.55/52.89    |  54.55/52.95    |  54.55/52.53    |  54.55/52.76    |   54.55/52.55   |    54.56/52.34  |   54.56/52.47   |  54.56/52.56    | 54.56/52.83 |  54.56/52.97    |  54.56/53.17    |  54.56/53.18    |  54.56/53.40    |    54.57/53.49   |   54.57/53.94    |       54.56/53.77    |  54.57/53.99    |   |  54.57/54.42    |
|  ACC(IN/OUT)  :arrow_down: |   19.18/15.40    | 19.18/5.12     |  19.18/3.10    |19.18/2.33| 19.18/1.96     |  19.18/1.77    |  19.18/1.64    |  19.18/1.57    |  19.18/1.51    | 19.18/1.47 |  19.18/1.43    |  19.18/1.41    |  19.18/1.39    |   19.18/1.38   |   19.18/1.36   |  19.18/1.36    |  19.18/1.36    |  19.18/1.39    |      |   19.18/1.39   |



### AIST SPIN 3D Pose

| Interval/Q | 1/100 | 2/50 | 3/33 | 4/25 | 5/20 | 6/16 | 7/14 | 8/12 | 9/11 | 10/10 | 11/9 | 12/8 | 13/7 | 14/7 | 15/6 | 16/6 | 17/5 | 18/5 | 19/5 | 20/5 |
| ---------- | ----- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ----- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- |
| MPJPE(IN/OUT) :arrow_down:    |   107.70/67.20    |   107.70/66.91   |   107.79/66.60   |  107.70/ 66.81    |   107.70/67.61   |  107.83/68.40    |  107.81/69.71    |  107.83/71.18    |  107.79/71.56    | 107.26/71.27 |  107.79/76.13    |   107.83/77.09   |   107.54/78.95   |  107.81/80.16    |  107.59/82.28    |  107.83/84.27    |  107.60/85.18    |  107.59/86.98    |   107.39/88.87   |   107.70/90.82   |
|  ACC(IN/OUT) :arrow_down:  |   33.82/7.58    |  33.82/7.87    |  33.88/7.61    |  33.82/6.87    |  33.82/6.58    |  33.93/6.31    |  33.91/6.09    |  33.93/5.97    |  33.88/5.89    | 33.37/5.68 |  33.88/5.70    |  33.93/5.64    |  33.67/5.61    |   33.91/5.53   |   33.72/5.54   |  33.93/5.48    |   33.72/5.47   |  33.72/5.40    |   33.50/5.30   |   33.82/5.32   |



### PW3D SPIN SMPL

| Interval/Q | 1/100 | 2/50 | 3/33 | 4/25 | 5/20 | 6/16 | 7/14 | 8/12 | 9/11 | 10/10 | 11/9 | 12/8 | 13/7 | 14/7 | 15/6 | 16/6 | 17/5 | 18/5 | 19/5 | 20/5 |
| ---------- | ----- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ----- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- |
| MPJPE(IN/OUT) :arrow_down:    |   100.13/96.80    |  100.13/96.36    |   100.03/95.95   |    100.13/96.18  |   100.13/96.10   |   100.24/96.38   |  100.29/97.02    |  100.24/97.47    |   100.03/97.42   |   100.13/97.53   |  100.03/97.93    |  100.24/99.34    |   100.24/99.76   |  100.29/100.49    |  100.30/101.02    |   100.24/101.78   |   100.12/102.13   |  100.30/103.19    |    100.11/102.46  |  100.13/104.21    |
|  ACC(IN/OUT) :arrow_down:  |   35.53/36.20    |   35.53/17.07   |   35.46/12.50   |  35.53/10.91    |  35.53/9.82    |  35.67/9.36    |   35.67/9.02    |  35.67/8.83    |  35.46/8.50    |  35.53/8.38     |  35.46/8.34    |  35.67/8.23    |  35.49/8.16    |   35.67/8.15   |  35.60/8.08    |  35.67/8.07    |   35.52/8.02   |  35.60/8.01    |  35.51/7.88    |  35.53/7.81    |




### PW3D EFT SMPL

| Interval/Q | 1/100 | 2/50 | 3/33 | 4/25 | 5/20 | 6/16 | 7/14 | 8/12 | 9/11 | 10/10 | 11/9 | 12/8 | 13/7 | 14/7 | 15/6 | 16/6 | 17/5 | 18/5 | 19/5 | 20/5 |
| ---------- | ----- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ----- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- |
| MPJPE(IN/OUT)  :arrow_down:   |  91.60/92.40     |  91.60/91.70    |  91.42/91.05    |  91.60/91.12    |  91.60/91.05    |  91.73/91.49    |   91.83/92.35   |  91.73/92.94    |   91.42/92.53   | 91.60/92.56   |  91.42/94.57    |  91.73/94.40    |  91.92/96.07    |  91.83/96.87    |  91.88/96.66    |  91.73/97.55   |  91.78/98.38    |   91.88/99.39   |   91.60/99.90   |  91.60/99.95    |
|  ACC(IN/OUT) :arrow_down:  |  33.57/33.07     |  33.57/16.32    | 33.51/12.30 |  33.57/10.87    |  33.57/10.05    |  33.87/9.64    |  33.86/9.37    |   33.87/9.05   |  33.51/8.79    | 33.57/8.75 |   33.51/8.64   |  33.87/8.62    |  33.75/8.49    |  33.86/8.55    |   33.71/8.42   |  33.87/8.43    |   33.81/8.31   |33.71/8.34|   33.68/8.20    |  33.57/8.22    |



### PW3D PARE SMPL

| Interval/Q | 1/100 | 2/50 | 3/33 | 4/25 | 5/20 | 6/16 | 7/14 | 8/12 | 9/11 | 10/10 | 11/9 | 12/8 | 13/7 | 14/7 | 15/6 | 16/6 | 17/5 | 18/5 | 19/5 | 20/5 |
| ---------- | ----- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ----- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- |
| MPJPE(IN/OUT)  :arrow_down:   |   80.44/81.05    |   80.44/80.58   |   80.38/80.29   |  80.44/80.49    |  80.44/80.71    |   80.64/81.24   |   80.65/81.60   |  80.64/82.05   |   80.38/82.25   | 80.44/81.76 |   80.38/83.87   |  80.64/84.58    |   80.54/85.67    |  80.65/86.70    |  80.60/86.90    |  80.64/87.99    |   80.55/88.63   |   80.60/89.58   |  80.47/90.35    |   80.44/91.20   |
|  ACC(IN/OUT) :arrow_down:  |  26.77/26.97     |   26.77/13.14   |  26.76/10.08    |  26.77/8.99    |  26.77/8.26    |  26.97/8.08    |   26.97/7.69    |  26.97/7.63    |  26.76/7.35    | 26.77/7.24 |  26.76/7.20    |  26.97/7.18    |   26.79/7.05    |  26.97/7.06  |  26.81/6.99    |  26.97/7.00    |   26.91/6.97   |   26.81/6.88   |  26.86/6.88    |  26.77/6.83    |



### AIST SPIN SMPL

| Interval/Q | 1/100 | 2/50 | 3/33 | 4/25 | 5/20 | 6/16 | 7/14 | 8/12 | 9/11 | 10/10 | 11/9 | 12/8 | 13/7 | 14/7 | 15/6 | 16/6 | 17/5 | 18/5 | 19/5 | 20/5 |
| ---------- | ----- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ----- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- |
| MPJPE(IN/OUT)  :arrow_down:   |   108.25/77.67    |   108.25/77.42   |   108.34/77.70   |   108.25/77.67   |  108.25/78.44    |   108.38/78.80   |  108.36/79.19    |   108.38/80.66   |  108.34/81.24    | 108.25/82.10 |   108.34/83.28   |  108.38/84.23    |  108.10/85.38    |  108.36/87.78    |   108.15/89.11   |  108.38/91.28    |   108.15/93.02   |   108.15/94.50   |   107.95/95.83   |  108.25/98.18    |
|  ACC(IN/OUT)  :arrow_down: |  33.83/13.46     | 33.83/10.95    |  33.90/10.00    |   33.83/8.87   |  33.83/8.15    |  33.95/7.84    |  33.93/7.61    |   33.95/7.55   |  33.90/7.40    | 33.83/7.27 |  33.90/7.18    |  33.95/7.16    |  33.69/7.11    |   33.93/7.13   |  33.74/7.10    |   33.95/7.10   |   33.73/7.11   |   33.74/7.00   |   33.52/6.97   |  33.83/6.97    |



