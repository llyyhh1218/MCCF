
<div align="center">
<h1> Mamba-based Cross-Modal Collaborative Fusion Network for PET-CT Lung Tumor Segmentation </h1>

## Environment

1. Create environment.
    ```shell
    conda create -n MIPA python=3.10
    conda activate MIPA
    ```

2. Install all dependencies.
Install pytorch, cuda and cudnn, then install other dependencies via:
    ```shell
    pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu118
   ```
    ```shell
    pip install -r requirements.txt
    ```

3. Install selective_scan_cuda_core.
    ```shell
    cd models/encoders/selective_scan
    pip install .
    cd ../../..
    ```

## Data Preparation

1. For our dataset PCLT20K, we orgnize the dataset folder in the following structure:
    ```shell
    <PCLT20K>
        |-- <0001>
            |-- <name1_CT.png>
            |-- <name1_PET.png>
            |-- <name1_mask.png>
            ...
        |-- <0002>
            |-- <name2_CT.png>
            |-- <name2_PET.png>
            |-- <name2_mask.png>
            ...
        ...
        |-- train.txt
        |-- test.txt
    ```

    `train.txt/test.txt` contains the names of items in training/testing set, e.g.:

    ```shell
    <name1>
    <name2>
    ...
    ```
2. Please put our dataset in the `data` directory

## Usage

### Training
1. Please download the pretrained [VMamba](https://github.com/MzeroMiko/VMamba) weights, and put them under `pretrained/vmamba/`. We use VMamba_Tiny as default.

    - [VMamba_Tiny](https://drive.google.com/file/d/1W0EFQHvX4Cl6krsAwzlR-VKqQxfWEdM8/view?usp=drive_link)
    - [VMamba_Small](https://drive.google.com/file/d/1671QXJ-faiNX4cYUlXxf8kCpAjeA4Oah/view?usp=drive_link)
    - [VMamba_Base](https://drive.google.com/file/d/1qdH-CQxyUFLq6hElxCANz19IoS-_Cm1L/view?usp=drive_link)


2. Config setting.

    Edit config in the `train.py`.
    Change C.backbone to `sigma_tiny` / `sigma_small` / `sigma_base` to use the three versions of VMamba.

3. Run multi-GPU distributed training:

    ```shell
    torchrun --nproc_per_node 'GPU_Numbers' train.py
    ```

4. You can also use single-GPU training:

    ```shell
    python train.py
    ```
5. Results will be saved in `save_model` folder.

