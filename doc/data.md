# Data Description

All the files mentioned below can be downloaded here. 

[Google Drive](https://drive.google.com/drive/folders/1e5wEPWFNldihU5mBUpTOuQaGjgIxujrt?usp=sharing)

[Baidu Netdisk](https://pan.baidu.com/s/1ZBgQDJElkObHBhLsWtmkQw?pwd=cqcw)

Valid data includes:


| Dataset | Pose Estimator | 3D Pose | 2D Pose | SMPL |
|  ----  | ----  | ----  | ----  | ----  |
| [Sub-JHMDB](http://jhmdb.is.tue.mpg.de/)  | [SimplePose](https://github.com/microsoft/human-pose-estimation.pytorch) |   | ✔ |   |
| [3DPW](https://virtualhumans.mpi-inf.mpg.de/3DPW/)  | [EFT](https://github.com/facebookresearch/eft) | ✔ |   |  ✔ |
| [3DPW](https://virtualhumans.mpi-inf.mpg.de/3DPW/)  | [PARE](https://pare.is.tue.mpg.de/) | ✔ |   | ✔ |
| [3DPW](https://virtualhumans.mpi-inf.mpg.de/3DPW/)  | [SPIN](https://github.com/nkolot/SPIN) | ✔ |   | ✔ |
| [Human3.6M](http://vision.imar.ro/human3.6m/description.php)  | [FCN](https://github.com/una-dinosauria/3d-pose-baseline) | ✔ |   |   |
| [AIST++](https://google.github.io/aistplusplus_dataset/factsfigures.html)  | [SPIN](https://github.com/nkolot/SPIN) | ✔ |   | ✔ |


All the models have the same settings with the original paper (e.g. training dataset and hyperparameters). There results are tested by us for fair comparison. We have make sure the dataset we test on have no overlap with the training dataset the model trained on. Specifically, we used architecture of '384x384_pose_resnet_101_d256d256d256' with trained weight on MPII for Simplepose.


If you want to add your own datasets, please 

- Organize the groundtruth data format following our settings(we recommend you to use the same format as follows) to generate ```\data\groundtruth_poses\[new_dataset]\[new_dataset]_gt_test.npz``` and ```\data\groundtruth_poses\[new_dataset]\[new_dataset]_gt_train.npz```. 

- Organize the detected data format following our settings(we recommend you to use the same format as follows) to generate ```\data\detected_poses\[new_dataset]\[estimator]\[new_dataset]_[estimator]_test.npz``` and ```\data\detected_poses\[new_dataset]\[estimator]\[new_dataset]_[estimator]_train.npz```.

- Add groundtruth data path, detected data path, keypoint number, and keypoint root in ```lib\core\config.py```.

- write ```\lib\core\dataset\[new_dataset]_dataset.py``` following the files under ```\lib\core\dataset\```.

**How to transfer custom data into our data?**:

- First, our 3d position is the root-relative 3d position in meter; 2d position is the normalized 2d pixel position in an image; SMPL parameters are the original outputs from estimators (e.g., PARE).

- To facilitate the transformation from raw output data into our data, we provide these transformation functions as follows.
  - For 2D pose transformation, if inputting the 2d positions under the pixel coordination, you can use [normalize_screen_coordinates](https://github.com/cure-lab/DeciWatch/blob/main/lib/utils/cam_utils.py#L11) to normalize the pixel-wise 2d position into [-1, 1], and then put them into the model for training and inference. Lastly, you can use [image_coordinates](https://github.com/cure-lab/DeciWatch/blob/main/lib/utils/cam_utils.py#L23) to denormalize the position into a pixel unit for error calculation and visualization.
  - For 3D pose transformation, if inputting the 3d positions under the world coordinate, you can use [world_to_camera](https://github.com/cure-lab/DeciWatch/blob/main/lib/utils/cam_utils.py#L97) and then subtract the root 3d position to get the root-relative 3d position in meter. We calculate the MPJPE and Accel under the root-relative 3d position in millimeter. Also, you can use [camera_to_world](https://github.com/cure-lab/DeciWatch/blob/main/lib/utils/cam_utils.py#L102) for visualization.
  - Besides, if you need to get the projected 2d positions from 3d positions under the camera coordinate, you can use [project_to_2d](https://github.com/cure-lab/DeciWatch/blob/main/lib/utils/cam_utils.py#L126) with distortion parameters or [project_to_2d_linear](https://github.com/cure-lab/DeciWatch/blob/main/lib/utils/cam_utils.py#L160) without distortion parameters.



## 3DPW

The sructure of the data should look like this:
```
|-- data
    |-- groundtruth_poses
        |-- pw3d 
            |-- pw3d_gt_test.npz
            |-- pw3d_gt_train.npz
        |-- ...
    |-- detected_poses
        |-- pw3d
            |-- spin
                |-- pw3d_spin_test.npz
                |-- pw3d_spin_train.npz
            |-- pare
                |-- pw3d_pare_test.npz
                |-- pw3d_pare_train.npz
            |-- eft
                |-- pw3d_eft_test.npz
                |-- pw3d_eft_train.npz
        |-- ...
    |-- checkpoints
    |-- smpl
    |-- videos
```


- ``pw3d_gt_test.npz``
            
    For ease of use, we processed the raw testing set of 3DPW dataset and re-stored the valid poses (campose_valid==1) in testing set.
            
    The .npz-file contains a dictionary with the following fields:
    
    - ``imgname``

        Strings containing the image and sequence name with format [sequence_name]/[image_name]. The length of the list is 37 and the order of the sequence is as follows. Duplicate sequence name means there are two person in one video sequence. There are 35515 frames in total. The order of parameter ``shape``, ``pose``, and ``joints_3d`` are the same with ``imgname``
        
        ```jsx
        downtown_enterShop_00
        flat_packBags_00
        downtown_walkBridge_01
        downtown_bus_00
        downtown_bus_00
        downtown_weeklyMarket_00
        downtown_walkUphill_00
        downtown_warmWelcome_00
        downtown_warmWelcome_00
        office_phoneCall_00
        office_phoneCall_00
        downtown_crossStreets_00
        downtown_crossStreets_00
        downtown_upstairs_00
        downtown_stairs_00
        downtown_walking_00
        downtown_walking_00
        downtown_downstairs_00
        downtown_car_00
        downtown_car_00
        flat_guitar_01
        downtown_arguing_00
        downtown_arguing_00
        downtown_runForBus_00
        downtown_runForBus_00
        downtown_rampAndStairs_00
        downtown_rampAndStairs_00
        downtown_windowShopping_00
        downtown_cafe_00
        downtown_cafe_00
        downtown_bar_00
        downtown_bar_00
        downtown_sitOnStairs_00
        downtown_sitOnStairs_00
        downtown_runForBus_01
        downtown_runForBus_01
        outdoors_fencing_01
        ```
        
    - ``shape``
        
        Ground_truth SMPL shape parameter. The shape of each sequence is corresponding_sequence_length\*10.
        
    - ``pose``
        
        Ground_truth SMPL pose parameter. The shape of each sequence is corresponding_sequence_length\*72.
    
    - ``joints_3d``

        Ground_truth 3D joint position. The shape of each sequence is corresponding_sequence_length\*(17\*3). Joints are in Human3.6M-format:
        
        ```jsx
        'hip',  # 0
        'lhip',  # 1
        'lknee',  # 2
        'lankle',  # 
        'rhip',  # 4
        'rknee',  # 5
        'rankle',  # 6
        'Spine (H36M)',  # 7
        'neck',  # 8
        'Head (H36M)',  # 9
        'headtop',  # 10
        'lshoulder',  # 11
        'lelbow',  # 12
        'lwrist',  # 13
        'rshoulder',  # 14
        'relbow',  # 15
        'rwrist',  # 16
        ```
        
- ``pw3d_gt_train.npz``
    
    For ease of use, we processed the raw training set of 3DPW dataset and re-stored the valid poses (campose_valid==1) in training set.
    
    The .npz-file contains a dictionary with the following fields:
    
    - ``imgname``
        
        Strings containing the image and sequence name with format [sequence_name]/[image_name]. The length of the list is 34 and the order of the sequence is as follows. Duplicate sequence name means there are two person in one video sequence. There are 22735 frames in total. The order of parameter ``shape``, ``pose``, and ``joints_3d`` are the same with ``imgname``
        
        ```jsx
        outdoors_freestyle_00
        courtyard_laceShoe_00
        courtyard_bodyScannerMotions_00
        courtyard_capoeira_00
        courtyard_capoeira_00
        courtyard_relaxOnBench_00
        courtyard_giveDirections_00
        courtyard_giveDirections_00
        courtyard_box_00
        outdoors_climbing_02
        outdoors_slalom_01
        courtyard_arguing_00
        courtyard_arguing_00
        outdoors_climbing_00
        courtyard_shakeHands_00
        courtyard_shakeHands_00
        courtyard_relaxOnBench_01
        courtyard_captureSelfies_00
        courtyard_captureSelfies_00
        courtyard_golf_00
        courtyard_backpack_00
        outdoors_climbing_01
        courtyard_goodNews_00
        courtyard_goodNews_00
        courtyard_rangeOfMotions_00
        courtyard_rangeOfMotions_00
        courtyard_dancing_01
        courtyard_dancing_01
        courtyard_basketball_00
        courtyard_basketball_00
        outdoors_slalom_00
        courtyard_jacket_00
        courtyard_warmWelcome_00
        courtyard_warmWelcome_00
        ```
        
    - ``shape``
        
        Ground_truth SMPL shape parameter. The shape of each sequence is corresponding_sequence_length\*10.
        
    - ``pose``
        
        Ground_truth SMPL pose parameter. The shape of each sequence is corresponding_sequence_length\*72.

    - ``joints_3d``

        Ground_truth 3D joint position. The shape of each sequence is corresponding_sequence_length\*(17\*3). Joints are in Human3.6M-format.
        
- ``pw3d_spin_test.npz``

    The .npz-file contains a dictionary with the following fields:

    - ``imgname``
        
        Same with pw3d_gt_test.npz
        
    - ``shape``
        
        The predicted SMPL shape parameter, with the same format as pw3d_gt_test.npz
        
    - ``pose``
        
        The predicted SMPL pose parameter, with the same format as pw3d_gt_test.npz
        
    - ``camera``
        
        The predicted camera parameter. The shape of each sequence is corresponding_sequence_length\*3.
        
    - ``joints_3d``
        
        The predicted 3D joint position, with the same format as pw3d_gt_test.npz
        
- ``pw3d_spin_train.npz``

    The .npz-file contains a dictionary with the following fields:

    - ``imgname``
        
        Same with pw3d_gt_train.npz
        
    - ``shape``
        
        The predicted SMPL shape parameter, with the same format as pw3d_gt_train.npz
        
    - ``pose``
        
        The predicted SMPL pose parameter, with the same format as pw3d_gt_train.npz
        
    - ``camera``
        
        The predicted camera parameter. The shape of each sequence is corresponding_sequence_length\*3.
        
    - ``joints_3d``
        
        The predicted 3D joint position, with the same format as pw3d_gt_train.npz

- ``pw3d_pare_test.npz``
    
    Same with pw3d_spin_test.npz

- ``pw3d_pare_train.npz``
    
    Same with pw3d_spin_train.npz

- ``pw3d_eft_test.npz``
    
    Same with pw3d_spin_test.npz

- ``pw3d_eft_train.npz``
    
    Same with pw3d_spin_train.npz



## Human3.6M

The sructure of the data should look like this:
```
|-- data
    |-- groundtruth_poses
        |-- h36m 
            |-- h36m_gt_test.npz
            |-- h36m_gt_train.npz
        |-- ...
    |-- detected_poses
        |-- h36m
            |-- fcn
                |-- h36m_fcn_test.npz
                |-- h36m_fcn_train.npz
        |-- ...

```

- ``h36m_gt_test.npz``
    
    For ease of use, we processed the raw testing set of Human3.6M dataset and re-stored the valid poses in testing set.
            
    The .npz-file contains a dictionary with the following fields:  

    - ``imgname``

        Strings containing the subject id, action name, camera id and image id with format S[subject_id]/[action_name]/camera[camera_id]/[image_id]. The length of the list is 236. There are 543344 frames in total. The order of parameter ``joints_3d`` is the same with ``imgname``. The camera parameters are the same order with the dictionary shown as follows.

        ```
        h36m_cameras_intrinsic_params = [
            {
                'id': '54138969',
                'center': [512.54150390625, 515.4514770507812],
                'focal_length': [1145.0494384765625, 1143.7811279296875],
                'radial_distortion': [-0.20709891617298126, 0.24777518212795258, -0.0030751503072679043],
                'tangential_distortion': [-0.0009756988729350269, -0.00142447161488235],
                'res_w': 1000,
                'res_h': 1002,
                'azimuth': 70,  # Only used for visualization
            },
            {
                'id': '55011271',
                'center': [508.8486328125, 508.0649108886719],
                'focal_length': [1149.6756591796875, 1147.5916748046875],
                'radial_distortion': [-0.1942136287689209, 0.2404085397720337, 0.006819975562393665],
                'tangential_distortion': [-0.0016190266469493508, -0.0027408944442868233],
                'res_w': 1000,
                'res_h': 1000,
                'azimuth': -70,  # Only used for visualization
            },
            {
                'id': '58860488',
                'center': [519.8158569335938, 501.40264892578125],
                'focal_length': [1149.1407470703125, 1148.7989501953125],
                'radial_distortion': [-0.2083381861448288, 0.25548800826072693, -0.0024604974314570427],
                'tangential_distortion': [0.0014843869721516967, -0.0007599993259645998],
                'res_w': 1000,
                'res_h': 1000,
                'azimuth': 110,  # Only used for visualization
            },
            {
                'id': '60457274',
                'center': [514.9682006835938, 501.88201904296875],
                'focal_length': [1145.5113525390625, 1144.77392578125],
                'radial_distortion': [-0.198384091258049, 0.21832367777824402, -0.008947807364165783],
                'tangential_distortion': [-0.0005872055771760643, -0.0018133620033040643],
                'res_w': 1000,
                'res_h': 1002,
                'azimuth': -110,  # Only used for visualization
            },
        ]

        h36m_cameras_extrinsic_params = {
            'S1': [
                {
                    'orientation': [0.1407056450843811, -0.1500701755285263, -0.755240797996521, 0.6223280429840088],
                    'translation': [1841.1070556640625, 4955.28466796875, 1563.4454345703125],
                },
                {
                    'orientation': [0.6157187819480896, -0.764836311340332, -0.14833825826644897, 0.11794740706682205],
                    'translation': [1761.278564453125, -5078.0068359375, 1606.2650146484375],
                },
                {
                    'orientation': [0.14651472866535187, -0.14647851884365082, 0.7653023600578308, -0.6094175577163696],
                    'translation': [-1846.7777099609375, 5215.04638671875, 1491.972412109375],
                },
                {
                    'orientation': [0.5834008455276489, -0.7853162288665771, 0.14548823237419128, -0.14749594032764435],
                    'translation': [-1794.7896728515625, -3722.698974609375, 1574.8927001953125],
                },
            ],
            'S5': [
                {
                    'orientation': [0.1467377245426178, -0.162370964884758, -0.7551892995834351, 0.6178938746452332],
                    'translation': [2097.3916015625, 4880.94482421875, 1605.732421875],
                },
                {
                    'orientation': [0.6159758567810059, -0.7626792192459106, -0.15728192031383514, 0.1189815029501915],
                    'translation': [2031.7008056640625, -5167.93310546875, 1612.923095703125],
                },
                {
                    'orientation': [0.14291371405124664, -0.12907841801643372, 0.7678384780883789, -0.6110143065452576],
                    'translation': [-1620.5948486328125, 5171.65869140625, 1496.43701171875],
                },
                {
                    'orientation': [0.5920479893684387, -0.7814217805862427, 0.1274748593568802, -0.15036417543888092],
                    'translation': [-1637.1737060546875, -3867.3173828125, 1547.033203125],
                },
            ],
            'S6': [
                {
                    'orientation': [0.1337897777557373, -0.15692396461963654, -0.7571090459823608, 0.6198879480361938],
                    'translation': [1935.4517822265625, 4950.24560546875, 1618.0838623046875],
                },
                {
                    'orientation': [0.6147197484970093, -0.7628812789916992, -0.16174767911434174, 0.11819244921207428],
                    'translation': [1969.803955078125, -5128.73876953125, 1632.77880859375],
                },
                {
                    'orientation': [0.1529948115348816, -0.13529130816459656, 0.7646096348762512, -0.6112781167030334],
                    'translation': [-1769.596435546875, 5185.361328125, 1476.993408203125],
                },
                {
                    'orientation': [0.5916101336479187, -0.7804774045944214, 0.12832270562648773, -0.1561593860387802],
                    'translation': [-1721.668701171875, -3884.13134765625, 1540.4879150390625],
                },
            ],
            'S7': [
                {
                    'orientation': [0.1435241848230362, -0.1631336808204651, -0.7548328638076782, 0.6188824772834778],
                    'translation': [1974.512939453125, 4926.3544921875, 1597.8326416015625],
                },
                {
                    'orientation': [0.6141672730445862, -0.7638262510299683, -0.1596645563840866, 0.1177929937839508],
                    'translation': [1937.0584716796875, -5119.7900390625, 1631.5665283203125],
                },
                {
                    'orientation': [0.14550060033798218, -0.12874816358089447, 0.7660516500473022, -0.6127139329910278],
                    'translation': [-1741.8111572265625, 5208.24951171875, 1464.8245849609375],
                },
                {
                    'orientation': [0.5912848114967346, -0.7821764349937439, 0.12445473670959473, -0.15196487307548523],
                    'translation': [-1734.7105712890625, -3832.42138671875, 1548.5830078125],
                },
            ],
            'S8': [
                {
                    'orientation': [0.14110587537288666, -0.15589867532253265, -0.7561917304992676, 0.619644045829773],
                    'translation': [2150.65185546875, 4896.1611328125, 1611.9046630859375],
                },
                {
                    'orientation': [0.6169601678848267, -0.7647668123245239, -0.14846350252628326, 0.11158157885074615],
                    'translation': [2219.965576171875, -5148.453125, 1613.0440673828125],
                },
                {
                    'orientation': [0.1471444070339203, -0.13377119600772858, 0.7670128345489502, -0.6100369691848755],
                    'translation': [-1571.2215576171875, 5137.0185546875, 1498.1761474609375],
                },
                {
                    'orientation': [0.5927824378013611, -0.7825870513916016, 0.12147816270589828, -0.14631995558738708],
                    'translation': [-1476.913330078125, -3896.7412109375, 1547.97216796875],
                },
            ],
            'S9': [
                {
                    'orientation': [0.15540587902069092, -0.15548215806484222, -0.7532095313072205, 0.6199594736099243],
                    'translation': [2044.45849609375, 4935.1171875, 1481.2275390625],
                },
                {
                    'orientation': [0.618784487247467, -0.7634735107421875, -0.14132238924503326, 0.11933968216180801],
                    'translation': [1990.959716796875, -5123.810546875, 1568.8048095703125],
                },
                {
                    'orientation': [0.13357827067375183, -0.1367100477218628, 0.7689454555511475, -0.6100738644599915],
                    'translation': [-1670.9921875, 5211.98583984375, 1528.387939453125],
                },
                {
                    'orientation': [0.5879399180412292, -0.7823407053947449, 0.1427614390850067, -0.14794869720935822],
                    'translation': [-1696.04345703125, -3827.099853515625, 1591.4127197265625],
                },
            ],
            'S11': [
                {
                    'orientation': [0.15232472121715546, -0.15442320704460144, -0.7547563314437866, 0.6191070079803467],
                    'translation': [2098.440185546875, 4926.5546875, 1500.278564453125],
                },
                {
                    'orientation': [0.6189449429512024, -0.7600917220115662, -0.15300633013248444, 0.1255258321762085],
                    'translation': [2083.182373046875, -4912.1728515625, 1561.07861328125],
                },
                {
                    'orientation': [0.14943228662014008, -0.15650227665901184, 0.7681233882904053, -0.6026304364204407],
                    'translation': [-1609.8153076171875, 5177.3359375, 1537.896728515625],
                },
                {
                    'orientation': [0.5894251465797424, -0.7818877100944519, 0.13991211354732513, -0.14715361595153809],
                    'translation': [-1590.738037109375, -3854.1689453125, 1578.017578125],
                },
            ],
        }

        ```

    - ``joints_3d``

        Ground_truth 3D joint position. The shape of each sequence is corresponding_sequence_length\*(17\*3). Joints are in Human3.6M-format.


- ``h36m_gt_train.npz``

    For ease of use, we processed the raw training set of Human3.6M dataset and re-stored the valid poses in training set.
            
    The .npz-file contains a dictionary with the following fields:  

    - ``imgname``

        Strings containing the subject id, action name, camera id and image id with format S[subject_id]/[action_name]/camera[camera_id]/[image_id]. The length of the list is 600. There are 1559752 frames in total. The order of parameter ``joints_3d`` is the same with ``imgname``.

     - ``joints_3d``

        Ground_truth 3D joint position. The shape of each sequence is corresponding_sequence_length\*(17\*3). Joints are in Human3.6M-format.


- h36m_fcn_test.npz

    - ``imgname``

        Same with h36m_gt_test.npz

    - ``joints_3d``

        Predicted 3D joint position. The shape of each sequence is corresponding_sequence_length\*(17\*3). Joints are in Human3.6M-format.


- h36m_fcn_train.npz

    - ``imgname``

        Same with h36m_gt_train.npz

    - ``joints_3d``

        Predicted 3D joint position. The shape of each sequence is corresponding_sequence_length\*(17\*3). Joints are in Human3.6M-format.



## AIST++

The sructure of the data should look like this:
```
|-- data
    |-- groundtruth_poses
        |-- aist
            |-- aist_gt_test.npz
            |-- aist_gt_train.npz
        |-- ...
    |-- detected_poses
        |-- aist
            |-- spin
                |-- aist_spin_test.npz
                |-- aist_spin_train.npz
        |-- ...

```

- ``aist_gt_test.npz``

    For ease of use, we processed the raw testing set of AIST++ dataset and re-stored the valid poses in testing set.
            
    The .npz-file contains a dictionary with the following fields:  

    - ``imgname``

        Strings containing the sequnce name and image id with format [sequence_name]/[image_id]. The length of the list is 3840. There are 2882640 frames in total. The order of parameter ``pose``, ``trans``, ``scaling``, ``joints_3d`` is the same with ``imgname``. 

    - ``pose``

        Ground_truth SMPL pose parameter. The shape of each sequence is corresponding_sequence_length\*72.

    - ``trans``

        Ground_truth motion 3D trajectory. The shape of each sequence is corresponding_sequence_length\*3.

    - ``scaling``

        Ground_truth human body scaling factor. A scalar value for each sequence. 

    - ``joints_3d``

        Ground_truth 3D joint position. The shape of each sequence is corresponding_sequence_length\*(14\*3). The order of the joints are as follows.
        ```
        "rankle",    # 0
        "rknee",     # 1 
        "rhip",      # 2 
        "lhip",      # 3 
        "lknee",     # 4 
        "lankle",    # 5  
        "rwrist",    # 6 
        "relbow",    # 7  
        "rshoulder", # 8  
        "lshoulder", # 9  
        "lelbow",    # 10  
        "lwrist",    # 11  
        "neck",      # 12  
        "headtop",   # 13  
        ```


- ``aist_gt_train.npz``

    For ease of use, we processed the raw training set of AIST++ dataset and re-stored the valid poses in training set.
            
    The .npz-file contains a dictionary with the following fields:  

    - ``imgname``

        Strings containing the sequnce name and image id with format [sequence_name]/[image_id]. The length of the list is 7292. There are 5916474 frames in total. The order of parameter ``pose``, ``trans``, ``scaling``, ``joints_3d`` is the same with ``imgname``. 

    - ``pose``

        Ground_truth SMPL pose parameter. The shape of each sequence is corresponding_sequence_length\*72.

    - ``trans``

        Ground_truth motion 3D trajectory. The shape of each sequence is corresponding_sequence_length\*3.

    - ``scaling``

        Ground_truth human body scaling factor. A scalar value for each sequence. 

    - ``joints_3d``

        Ground_truth 3D joint position. The shape of each sequence is corresponding_sequence_length\*(14\*3). The order of the joints are the same as aist_gt_test.npz.

- ``aist_spin_test.npz``


    The .npz-file contains a dictionary with the following fields:

    - ``imgname``
        
        Same with aist_gt_test.npz
        
    - ``shape``
        
        The predicted SMPL shape parameter.
        
    - ``pose``
        
        The predicted SMPL pose parameter, with the same format as aist_gt_test.npz
        
    - ``camera``
        
        The predicted camera parameter. The shape of each sequence is corresponding_sequence_length\*3.
        
    - ``joints_3d``
        
        The predicted 3D joint position, with the same format as aist_gt_test.npz

- ``aist_spin_train.npz``

    The .npz-file contains a dictionary with the following fields:

    - ``imgname``
        
        Same with aist_gt_train.npz
        
    - ``shape``
        
        The predicted SMPL shape parameter.
        
    - ``pose``
        
        The predicted SMPL pose parameter, with the same format as aist_gt_train.npz
        
    - ``camera``
        
        The predicted camera parameter. The shape of each sequence is corresponding_sequence_length\*3.
        
    - ``joints_3d``
        
        The predicted 3D joint position, with the same format as aist_gt_train.npz

## Sub-JHMDB

The sructure of the data should look like this:
```
|-- data
    |-- groundtruth_poses
        |-- jhmdb
            |-- jhmdb_gt_test.npz
            |-- jhmdb_gt_train.npz
        |-- ...
    |-- detected_poses
        |-- jhmdb
            |-- simplepose
                |-- jhmdb_simplepose_test.npz
                |-- jhmdb_simplepose_train.npz
        |-- ...

```

- ``jhmdb_gt_test.npz``

    For ease of use, we processed the raw testing set of Sub-JHMDB dataset and re-stored the valid poses in testing set.
            
    The .npz-file contains a dictionary with the following fields:  

    - ``imgname``

        Strings containing the action name, sequnce name and image id with format [action_name]/[sequence_name]/[image_id]. The length of the list is 261. There are 9228 frames in total. The order of parameter ``joints_2d`` is the same with ``imgname``. 

    - ``joints_2d``

        Ground_truth 2D joint position. The shape of each sequence is corresponding_sequence_length\*(15\*2). The order of the joints are as follows.

        ```
        1: neck
        2: belly
        3: face
        4: right shoulder
        5: left  shoulder
        6: right hip
        7: left  hip
        8: right elbow
        9: left elbow
        10: right knee
        11: left knee
        12: right wrist
        13: left wrist
        14: right ankle
        15: left ankle
        ```


- ``jhmdb_gt_train.npz``

    For ease of use, we processed the raw training set of Sub-JHMDB dataset and re-stored the valid poses in training set.
            
    The .npz-file contains a dictionary with the following fields:  

    - ``imgname``

        Strings containing the action name, sequnce name and image id with format [action_name]/[sequence_name]/[image_id]. The length of the list is 687. There are 24372 frames in total. The order of parameter ``joints_2d`` is the same with ``imgname``. 

    - ``joints_2d``

        Ground_truth 2D joint position. The shape of each sequence is corresponding_sequence_length\*(15\*2).

- ``jhmdb_simplepose_test.npz``

    The .npz-file contains a dictionary with the following fields:  

    - ``imgname``

        Same with jhmdb_gt_test.npz

    - ``joints_2d``

        Predicted 2D joint position. The shape of each sequence is corresponding_sequence_length\*(15\*2).

- ``jhmdb_simplepose_train.npz``

    The .npz-file contains a dictionary with the following fields:  

    - ``imgname``

        Same with jhmdb_gt_train.npz

    - ``joints_2d``

        Predicted 2D joint position. The shape of each sequence is corresponding_sequence_length\*(15\*2).







