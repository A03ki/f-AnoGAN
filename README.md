# f-AnoGAN

f-AnoGAN is a GAN for anomaly detection. One of the features of this GAN is that two adversarial networks (Generator and Discriminator) and Encoder are trained separately. In addition, an anomaly score is computed by both a discriminator feature residual error and an image reconstruction error.

## References
Papar

- [f-AnoGAN: Fast unsupervised anomaly detection with generative adversarial networks](https://www.sciencedirect.com/science/article/pii/S1361841518302640)

Github

- [tSchlegl/f-AnoGAN: Code for reproducing f-AnoGAN training and anomaly scoring](https://github.com/tSchlegl/f-AnoGAN)
- [PyTorch-GAN/wgan_gp.py at master · eriklindernoren/PyTorch-GAN](https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/wgan_gp/wgan_gp.py)

`fanogan/model.py`, `fanogan/train_wgangp.py` and `fanogan/train_encoder_izif.py` are modified eriklindernoren's `wgan_gp.py` for f-AnoGAN.

## Requirements

Python 3.6 or later  
PyTorch 1.x

Matplotlib  
Numpy  
pandas  
scikit-learn

## Usage

Please run below in order.

### Step: 1

```
python train_wgangp.py --training_label 1
```

### Step: 2

```
python train_encoder_izif.py --training_label 1
```

### Step: 3

```
python test_anomaly_detection.py --training_label 1
```

After Step: 3, `score.csv` will be generated in the directory `results`.

See `visualization.ipynb` about data visualization for `score.csv`.


## Colaboratory

[f-AnoGAN.ipynb](https://colab.research.google.com/drive/1mnuMH2gZH5RR47haP9r8Rv568G1mjJ1T)
