# Speech Deepfake Detection
This repository provides code for basic speech deepfake detection.


## Environment Setup

Clone the specific version of Fairseq required for this project:
[Fairseq Repository](https://github.com/pytorch/fairseq/tree/a54021305d6b3c4c5959ac9395135f63202db8f1)

Install Fairseq in editable mode:
```bash
cd fairseq
pip install --editable ./
```

Install all required dependencies:
```bash
pip install -r requirements.txt
```

## Data Preparation

### Audio and Metadata
Refer to the `data/README.md` for detailed instructions on preparing the audio folder structure and corresponding CSV metadata files.

### Pre-trained SSL Models
To use self-supervised learning (SSL) models, refer to the `pretrained/README.md` for instructions on downloading example checkpoints.


## Training and Evaluation

### Training
Train the model on a single or multiple GPUs:
```bash
python train.py --config configs/config.yaml --exp_dir exp/debug
```

### Evaluation
Evaluate the trained model on a single GPU:
```bash
python evaluate.py --exp_dir exp/debug --epoch best --eval 19LA
```


## References

This codebase is inspired by and adapted from the following repositories:
- [SSL Anti-spoofing](https://github.com/TakHemlata/SSL_Anti-spoofing/tree/main)
- [Clova AI AASIST](https://github.com/clovaai/aasist)
- [ASVspoof Challenge 2021](https://github.com/asvspoof-challenge/2021)
