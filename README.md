<p align="center">

  <h2 align="center">CES25: Image to 3D Mesh Reconstruction </h2>
  <p align="center">
    <br>
    <sup>1</sup>Polygome &nbsp;&nbsp;&nbsp; <sup>2</sup>Korea Electronics Technology Institute   &nbsp;&nbsp;&nbsp;
    <br>
    </br>
  </p>
    </p>

<div align="left">
  <br>
  This repository will contain the official implementation of <strong>2D to 3D HumanRecon</strong>.
</div>


## News & TODOs
- [ ] **[2025.01.xx]** Release inference code and pretrained weights
- [ ] **[2025.xx.xx]** Release xxx

## Models

|Model        | Resolution|#Views    |GPU Memery<br>(w/ refinement)|#Training Scans|Datasets|
|:-----------:|:---------:|:--------:|:--------:|:--------:|:--------:|
|unet_uv      |512x512    |-         |-GB    |~2500     |[THuman2.1](https://github.com/ytrock/THuman2.0-Dataset)|
|unet_color   |1024x1024  |2         |-GB    |~5500     |[THuman2.1](https://github.com/ytrock/THuman2.0-Dataset), [2K2K](https://github.com/SangHunHan92/2K2K)|

```
|--- ckpt/
|    |--- pretrained_weights/
|    |--- unet_uv/ or unet_color/
```

### Installation
```bash
# Create conda environment
conda create -n ces25 python=3.10
conda activate ces25

# Install PyTorch and other dependencies
conda install pytorch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 pytorch-cuda=12.4 -c pytorch -c nvidia
pip install -r requirements.txt

# detectron2
cd libs/detectron2
python setup.py install

# densepose
cd libs/detectron2/projects/DensePose
python setup.py install

# uvconverter
pip install UVTextureConverter

# nvdiffrast
git clone https://github.com/NVlabs/nvdiffrast.git
cd nvdiffrast
python setup.py install


```



