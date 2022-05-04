# Training Detail

Training configs are explained in [lib/core/config.py](../lib/core/config.py). Different configs for different datasets and estimators are in [config](../configs/) folder.

| Dataset | Pose Estimator | 3D Pose | 2D Pose | SMPL |
|  ----  | ----  | ----  | ----  | ----  |
| [Sub-JHMDB](http://jhmdb.is.tue.mpg.de/)  | [SimplePose](https://openaccess.thecvf.com/content_ECCV_2018/html/Bin_Xiao_Simple_Baselines_for_ECCV_2018_paper.html) |   | [config](../configs/config_jhmdb_simplepose_2D.yaml) |   |
| [3DPW](https://virtualhumans.mpi-inf.mpg.de/3DPW/)  | [EFT](https://github.com/facebookresearch/eft) | [config](../configs/config_pw3d_eft_3D.yaml) |   |  [config](../configs/config_pw3d_eft_smpl.yaml) |
| [3DPW](https://virtualhumans.mpi-inf.mpg.de/3DPW/)  | [PARE](https://pare.is.tue.mpg.de/) | [config](../configs/config_pw3d_pare_3D.yaml) |   | [config](../configs/config_pw3d_pare_smpl.yaml) |
| [3DPW](https://virtualhumans.mpi-inf.mpg.de/3DPW/)  | [SPIN](https://github.com/nkolot/SPIN) | [config](../configs/config_pw3d_spin_3D.yaml) |   | [config](../configs/config_pw3d_spin_smpl.yaml) |
| [Human3.6M](http://vision.imar.ro/human3.6m/description.php)  | [FCN](https://github.com/una-dinosauria/3d-pose-baseline) | [config](../configs/config_h36m_fcn_3D.yaml) |   |   |
| [AIST++](https://google.github.io/aistplusplus_dataset/factsfigures.html)  | [SPIN](https://github.com/nkolot/SPIN) | [config](../configs/config_aist_spin_3D.yaml) |   | [config](../configs/config_aist_spin_smpl.yaml) |


## Training Commands

You can directly train the model in different datasets and estimator settings using following commands


### 2D Pose
Sub-JHMDB Simplepose
```shell script
python train.py --cfg configs/config_jhmdb_simplepose_2D.yaml --dataset_name jhmdb --estimator simplepose --body_representation 2D --sample_interval 10

```

### 3D Pose
3DPW SPIN
```shell script
python train.py --cfg configs/config_pw3d_spin_3D.yaml --dataset_name pw3d --estimator spin --body_representation 3D --sample_interval 10

```
3DPW EFT
```shell script
python train.py --cfg configs/config_pw3d_eft_3D.yaml --dataset_name pw3d --estimator eft --body_representation 3D --sample_interval 10

```
3DPW PARE
```shell script
python train.py --cfg configs/config_pw3d_pare_3D.yaml --dataset_name pw3d --estimator pare --body_representation 3D --sample_interval 10

```
AIST++ SPIN
```shell script
python train.py --cfg configs/config_aist_spin_3D.yaml --dataset_name aist --estimator spin --body_representation 3D --sample_interval 10

```
Human3.6M FCN
```shell script
python train.py --cfg configs/config_h36m_fcn_3D.yaml --dataset_name h36m --estimator fcn --body_representation 3D --sample_interval 10

```

### SMPL
3DPW SPIN
```shell script
python train.py --cfg configs/config_pw3d_spin_smpl.yaml --dataset_name pw3d --estimator spin --body_representation smpl --sample_interval 10

```
3DPW EFT
```shell script
python train.py --cfg configs/config_pw3d_eft_smpl.yaml --dataset_name pw3d --estimator eft --body_representation smpl --sample_interval 10

```
3DPW PARE
```shell script
python train.py --cfg configs/config_pw3d_pare_smpl.yaml --dataset_name pw3d --estimator pare --body_representation smpl --sample_interval 10

```
AIST++ SPIN
```shell script
python train.py --cfg configs/config_aist_spin_smpl.yaml --dataset_name aist --estimator spin --body_representation smpl --sample_interval 10

```

## Useful configs

- Set ```cfg.TRAIN.RESUME = [checkpoint path]```, then you can resume training

- Set ```cfg.EXP_NAME = [your experiment name]```, then all the results would save in folder```[time]_[your experiment name]```
