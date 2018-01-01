# Variational Autoencoder

Variational Autoencoder implementation with Tensorflow.

<img src="https://s10.postimg.org/76n9tlbvd/faces.png" width="800">
<img src="https://s10.postimg.org/sgaw4hhw9/net.png" width="300">
<img src="https://s10.postimg.org/3n1c3swax/encoder1.png" width="400">
<img src="https://s10.postimg.org/lpuev0zvd/encoder2.png" width="500">
<img src="https://s10.postimg.org/64d3b3y7t/variance.png" width="400">
<img src="https://s10.postimg.org/fc5brrfjt/decoder.png" width="700">
<img src="https://s10.postimg.org/dkccwvjc9/loss.png" width="600">
<img src="https://s10.postimg.org/bfrzvtckp/optimizer.png" width="400">

## Usage
Prepare folder with images of fixed size.
```
git clone https://github.com/WojciechMormul/van.git
cd vae
python make_record.py --images-path=./data --record-path=./train.record
python vae_train.py
python vae_eval.py
```
