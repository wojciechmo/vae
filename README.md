# Variational Autoencoder

Variational Autoencoder implementation with Tensorflow.

<img src="https://github.com/WojciechMormul/vae/blob/master/imgs/faces2.png" width="800">

## Usage
Prepare folder with images of fixed size.
```
git clone https://github.com/WojciechMormul/van.git
cd vae
python make_record.py --images-path=./data --record-path=./train.record
python vae_train.py
python vae_eval.py
```
