<p align="center">

  <h2 align="center">HumanRecon: UV texture map and Displacement Prediction based Iterative Refinement </h2>
  <p align="center">
    <strong>Jumi Kang</strong></a><sup>1, 2</sup>
    · 
    <strong>Mingyu Park</strong></a><sup>1, 2</sup>
    · 
    <br>
    <sup>1</sup>Korea Electronics Technology Institute  &nbsp;&nbsp;&nbsp; <sup>2</sup>Polygome &nbsp;&nbsp;&nbsp;
    <br>
    </br>
  </p>
    </p>
<div align="center">
  <img src="./assets/teaser.png" alt="HumanRecon: UV texture map and Displacement Prediction based Iterative Refinement" style="width: 80%; height: auto;"></a>
</div>

<div align="left">
  Figure 1. Given a reference human image ~
</div>


<div align="left">
  <br>
  This repository will contain the official implementation of <strong>HumanRecon</strong>.
</div>


## 📣 News & TODOs
- [ ] **[2025.01.xx]** Release inference code and pretrained weights
- [ ] **[2025.xx.xx]** Release paper and project page
- [ ] Release reconstruction code.
- [ ] Release training code.

## 🧰 Models

|Model        | Resolution|#Views    |GPU Memery<br>(w/ refinement)|#Training Scans|Datasets|
|:-----------:|:---------:|:--------:|:--------:|:--------:|:--------:|
|hr_v1        |512x512    |4         |10.0GB    |~2500     |[THuman2.1](https://github.com/ytrock/THuman2.0-Dataset)|
|hr_v2        |512x512    |8         |20.0GB    |~5500     |[THuman2.1](https://github.com/ytrock/THuman2.0-Dataset), [2K2K](https://github.com/SangHunHan92/2K2K)|

```
|--- ckpt/
|    |--- pretrained_weights/
|    |--- hr_v1/ or hr_v2/
```


## ⚙️ Setup
### 1. Clone MagicMan
```bash
git clone https://github.com/hsil/HumanRecon.git
cd HumanRecon
```

### 2. Installation
```bash
# Create conda environment
conda create -n humanRecon python=3.10
conda activate humanRecon

# Install PyTorch and other dependencies
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2
pip install -r requirements.txt

```

## 💫 Inference
```

```

## 🙏 Acknowledgments


## ✏️ Citing
Please consider citing:
```BibTeX

```
