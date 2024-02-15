# ZeroShape

[[Project Page]](https://zixuanh.com/projects/zeroshape.html)  [[Paper]](https://zixuanh.com/projects/zeroshape/paper.pdf) [[Demo]](https://huggingface.co/spaces/zxhuang1698/ZeroShape)

<img src="teaser.gif" width="100%"/>

The repository currently includes demo, training, evaluation code and data for ZeroShape. 

## Dependencies

If your GPU supports CUDA 10.2, please install the dependencies by running
```bash
conda env create --file requirements.yaml
```

If you need newer CUDA versions, please install the dependencies manually:
```bash
conda create -n zeroshape python=3 pytorch::pytorch=1.11 pytorch::torchvision=0.12 cudatoolkit=10.2 (change these to your desired version)
conda install -c conda-forge tqdm pyyaml pip matplotlib trimesh tensorboard
pip install pyrender opencv-python pymcubes ninja timm
```

To use the segmentation preprocessing tool, please run
```bash
pip install rembg
```

## Demo
Please download the pretrained weights for shape reconstruction at [this url](https://www.dropbox.com/scl/fi/hv3w9z59dqytievwviko4/shape.ckpt?rlkey=a2gut89kavrldmnt8b3df92oi&dl=0) and place it under the `weights` folder. We have prepared some images and masks under the `examples` folder. To reconstruct their shape, please run:
```bash
python demo.py --yaml=options/shape.yaml --task=shape --datadir=examples --eval.vox_res=128 --ckpt=weights/shape.ckpt
```
The results will be saved under the `examples/preds` folder. 

To run the demo on your own images and masks, feel free to drop them in the `examples` folder. If you do not have mask, please run:
```bash
python preprocess.py path-to-your-image
```
The preprocessed image and mask will be saved in the `my_examples` folder. To reconstruct their shape, please run:
```bash
python demo.py --yaml=options/shape.yaml --task=shape --datadir=my_examples --eval.vox_res=128 --ckpt=weights/shape.ckpt
```

If you want to estimate the visible surface (depth and intrinsics), please download the pretrained weights for visible surface estimation at [this url](https://www.dropbox.com/scl/fi/1456be9dcwpwarrtgotny/depth.ckpt?rlkey=cmb3e76mw4dskomb0i51e99qt&dl=0) and place it under the `weights` folder. Then run:
```bash
python demo.py --yaml=options/depth.yaml --task=depth --datadir=examples --ckpt=weights/depth.ckpt
```
The results will be saved under the `examples/preds` folder.


## Data
Please download our curated training and evaluation data at the following links:
| Data| Link |
|-----|------|
| Training Data | [this url](https://www.dropbox.com/scl/fi/ac775o7n6wzzwvbw66cae/combined_synthetic.tar.gz?rlkey=c635jixlul9of8un2aw43xpq7&dl=0)   |
| OmniObject3D | [this url](https://www.dropbox.com/scl/fi/mnballhtedu71ggt8h8a9/OmniObject3D.tar?rlkey=wrsssq79y0p609xwyvege09na&dl=0) |
| Ocrtoc | [this url](https://www.dropbox.com/scl/fi/ltlwknufxugkpxz7q0ea6/Ocrtoc.tar?rlkey=omnzjrwwyi70fjla4e1nkmxlj&dl=0)   |
| Pix3D | [this url](https://www.dropbox.com/scl/fi/0vzx5b78pu1vh7z4w80ka/Pix3D.tar?rlkey=h3r7ihtxzgkjya0ul1psf7uj8&dl=0)   |

After extracting the data, organize your `data` folder as follows:

```
data
├── train_data/
|   ├── objaverse_LVIS/
|   |   ├── images_processed/
|   |   ├── lists/
|   |   ├── ...
|   ├── ShapeNet55/
|   |   ├── images_processed/
|   |   ├── lists/
|   |   ├── ...
├── OmniObject3D/
|   ├── images_processed/
|   ├── lists/
|   ├── ...
├── Ocrtoc/
|   ├── images_processed/
|   ├── lists/
|   ├── ...
├── Pix3D/
|   ├── img_processed/
|   ├── lists/
|   ├── ...
├── ...
```
Note that you do not have to download all the data. For example, if you only want to perform evaluation on one of the data source, feel free to only download and organize that specific one accordingly.

## Training

The first step of training ZeroShape is to pretrain the depth and intrinsics estimator. If you have downloaded the weights already (see demo), you can skip this step and use our pretrained weights at `weights/depth.ckpt`. If you want to train everything from scratch yourself, please run 
```bash
python train.py --yaml=options/depth.yaml --name=run-depth
```
The visualization and results will be saved at `output/depth/run-depth`. Once the training is finished, copy the weights from `output/depth/run-depth/best.ckpt` to `weights/depth.ckpt`.

To train the full reconstruction model, please run
```bash
python train.py --yaml=options/shape.yaml --name=run-shape
```
The visualization and results will be saved at `output/shape/run-shape`. 

## Evaluating

To evaluate the model on a specific test set (`omniobj3d|ocrtoc|pix3d`), please run

```bash
python evaluate.py --yaml=options/shape.yaml --name=run-shape --data.dataset_test=name_of_test_set --eval.vox_res=128 --eval.brute_force --eval.batch_size=1 --resume
```
The evaluation results will be printed and saved at `output/depth/run-shape`. If you want to evaluate the checkpoint we provided instead, feel free to create an empty folder `output/shape/run-shape` and move `weights/shape.ckpt` to `output/shape/run-shape/best.ckpt`

## References

If you find our work helpful, please consider citing our paper.
```
@article{huang2023zeroshape,
  title={ZeroShape: Regression-based Zero-shot Shape Reconstruction},
  author={Huang, Zixuan and Stojanov, Stefan and Thai, Anh and Jampani, Varun and Rehg, James M},
  journal={arXiv preprint arXiv:2312.14198},
  year={2023}
}
```
