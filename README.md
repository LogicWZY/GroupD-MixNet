# GroupD-MixNet: Accurate ultrasound lesion segmentation with multi-scale dynamic kernels

Official pytorch code base for Biomedical Signal Processing and Control: "GroupD-MixNet: Accurate ultrasound lesion segmentation with multi-scale dynamic kernels"



## Datasets

Please put the BUSI dataset or your own dataset as the following architecture. 
```
└── CMUNeXt
    ├── Dataset
        ├── BUSI
            ├── image
            |   ├── benign (10).png
            │   ├── malignant (17).png
            │   ├── ...
            |
            └── GT
                ├── benign (10).png
                ├── malignant (17).png
                ├── ...
            ├── BUSI_test_1.txt
            ├── BUSI_train_1.txt
        ├── your dataset
            ├── images
            |   ├── 0a7e06.png
            │   ├── ...
            |
            └── masks
                ├── 0
                |   ├── 0a7e06.png
                |   ├── ...
        ├── dataset.py
    ├── lib
    ├── log
    ├── utils
    └── main.py
```
## Environment

- GPU: NVIDIA GeForce RTX4090 GPU
- Pytorch: 1.13.0 cuda 11.7
- Python: 3.9
- pysodmetrics: 1.4.2

## Training and Validation

Follow the split setting in Dataset\BUSI:

Then, train and validate your dataset:

```python
python main.py --model GroupD_Mix --Deep_Supervision --flod 1 --early_stopping 70
```

## Acknowledgements:

This code-base uses helper functions from [CMU](https://github.com/FengheTan9/CMU-Net).

## Citation

If you use our code, please cite our paper:


```tex
@article{tang2023cmunext,
  title={GroupD-MixNet: Accurate ultrasound lesion segmentation with multi-scale dynamic kernels},
  author={Yuhao Wang and Wenyu Zhang and Chengwei Zhang and Fuxiang Lu},
  journal={Biomedical Signal Processing and Control},
  volume = {121},
  pages = {110248},
  year = {2026},
}
```
