# Variational Autoencoder

Variational Autoencoder implementation with Tensorflow.

<img src="https://github.com/WojciechMormul/vae/blob/master/imgs/faces2.png" width="800">
<img src="https://github.com/WojciechMormul/vae/blob/master/imgs/net.png" width="300">
<img src="https://github.com/WojciechMormul/vae/blob/master/imgs/encoder1.png" width="400">
<img src="https://github.com/WojciechMormul/vae/blob/master/imgs/encoder2.png" width="500">
<img src="https://github.com/WojciechMormul/vae/blob/master/imgs/variance.png" width="400">
<img src="https://github.com/WojciechMormul/vae/blob/master/imgs/decoder.png" width="700">
<img src="https://github.com/WojciechMormul/vae/blob/master/imgs/loss.png" width="600">

## Usage
Prepare folder with images of fixed size.
```
git clone https://github.com/WojciechMormul/van.git
cd vae
python make_record.py --images-path=./data --record-path=./train.record
python vae_train.py
python vae_eval.py
```
