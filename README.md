# RAAS

[ICLR 2026] Official Implementation for "Characterizing and Mitigating Reasoning Drift in Large Language Models"

## 1. Install

```bash
conda create -n raas python=3.10 
conda activate raas 

pip install -r requirements.txt 
```

## 2. Usage

```bash
cd ./distill_bert
bash run_script.sh

cd ../
python generate_vectors.py
python generate_anchors.py
python inference.py --dataset [DATASET] --model_path [MODEL_PATH] --method steering --insert_layer [LAYER]
```

## 3. Citation

```bash
@inproceedings{zhang2026characterizing,
title={Characterizing and mitigating reasoning drift in large language models},
author={Zhang, Yufeng and Wang, Xuepeng and Wu, Lingxiang and Wang, Jinqiao},
booktitle={The Fourteenth International Conference on Learning Representations},
year={2026}
}
```
