# Small-Incision Cataract Surgery Tool Segmentation

### To setup the environment

1. Create a new conda environment and activate it.
```bash
conda create -n toolseg python=3.12
conda activate toolseg
```

2. Install the required packages in the new environment.
```bash
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
pip install -r requirements.txt
```


### To download the dataset

1. Download the Sankara-MSICS dataset from [here](https://huggingface.co/datasets/SankaraEyeHospital/SankaraMSICS).

### To train the model

Make appropriate changes in `config.yaml` and run the following.
```bash
python main.py
```

Alternatively, you can override the parameters on the command line.
```bash
python main.py fold=0 condition.phase='pcd'
```

### To infer results based on pre-trained weights

Ensure the correct parameters are added in `config.yaml`.
```bash
python inference.py 
```