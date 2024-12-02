<p align="center">

  <h2 align="center">CES_2025: Image to 3D Reconstuction & Animation </h2>
  <p align="center">
    <br>
    <sup>1</sup>Polygom &nbsp;&nbsp;&nbsp; <sup>2</sup>Korea Electronics Technology Institute &nbsp;&nbsp;&nbsp;
    <br>
    </br>
  </p>
    </p>

<div align="left">
  <br>
  This repository will contain the official implementation of <strong>CES2025</strong>.
</div>


## News & TODOs
- [ ] **[2025.01.xx]** Release inference code and pretrained weights
- [ ] **[2025.xx.xx]** Release paper and project page

## Models

|Model        | Resolution|#Views    |GPU Memery<br>(w/ refinement)|#Training Scans|Datasets|
|:-----------:|:---------:|:--------:|:--------:|:--------:|:--------:|
|unet_uv      |512x512    |-         |10.0GB    |~2500     |[THuman2.1](https://github.com/ytrock/THuman2.0-Dataset)|
|unet_color   |1024x1024    |2         |20.0GB    |~5500     |[THuman2.1](https://github.com/ytrock/THuman2.0-Dataset), [2K2K](https://github.com/SangHunHan92/2K2K)|

```
|--- ckpt/
|    |--- predtrained/
|    |--- unet_uv/ or unet_color/
```

## Installation
```bash
# Create conda environment
conda create -n ces25 python=3.11
conda activate ces25

# Install PyTorch and other dependencies
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2
pip install -r requirements.txt

```

