import tensorflow as tf
import numpy as np
import scipy
import cv2

LATENT_DIM = 10
ROWS, COLS = 5, 12
HEIGHT, WIDTH, DEPTH = 144, 112, 3
N,M = 100, 30

splines=[]
x = range(N)
xs = np.linspace(0, N, N*M)

for i in range(ROWS*COLS*LATENT_DIM):
	
	y = np.random.normal(0.0, 1.0, size=[N]).astype(np.float32)
	s = scipy.interpolate.UnivariateSpline(x, y, s=2)
	ys = s(xs)
	splines.append(ys)

splines=np.array(splines)

with tf.Session() as sess: 

	saver = tf.train.import_meta_graph('./model/vae-200000.meta')
	saver.restore(sess,tf.train.latest_checkpoint('./model'))
	graph = tf.get_default_graph()
	latent_input = graph.get_tensor_by_name('latent_input:0')
	image_eval = graph.get_tensor_by_name('decoder/eval/conv5/act:0')
	
	for i in range(N*M):
		
		time_point = splines[...,i]
		time_point = np.reshape(time_point, [ROWS*COLS, LATENT_DIM])
		data = sess.run(image_eval, feed_dict = {latent_input: time_point})
		data = np.reshape(data, (ROWS, COLS, HEIGHT, WIDTH, DEPTH))
		data = np.concatenate(np.concatenate(data, 1), 1)
		cv2.imshow('eval_img', data)
		cv2.moveWindow('eval_img',0,0)
		key = cv2.waitKey(50)
		if key == 27:
			break
