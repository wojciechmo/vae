import tensorflow as tf
import numpy as np
import cv2
import os

LATENT_DIM = 10
HEIGHT, WIDTH, DEPTH = 144, 112, 3
H1, W1, D1, D2, D3, D4 = 9, 7, 16, 32, 64, 128
BATCH_SIZE, LEARNING_RATE, TRAIN_ITERS = 20, 1e-4, 200000
EVAL_ROWS, EVAL_COLS, SAMPLES_PATH, EVAL_INTERVAL = 8, 12, './samples', 500
MODEL_PATH, SAVE_INTERVAL = './model', 10000 

# -------------------------------------------------------------
# -------------------------- encoder --------------------------
# -------------------------------------------------------------

def lrelu(x, leak, name):
 
	return tf.maximum(x, leak*x, name=name)

def conv_layer(x, out_depth, kernel, strides, w_initializer, b_initializer, name):

	in_depth = x.get_shape()[3]

	with tf.name_scope(name): 

		w = tf.get_variable('w',shape=[kernel[0], kernel[1], in_depth, out_depth], initializer=w_initializer)
		b = tf.get_variable('b',shape=[out_depth], initializer=b_initializer)
		
		conv = tf.nn.conv2d(x, filter=w, strides=[1, strides[0], strides[1], 1], padding="SAME", name='conv')
		conv = tf.add(conv, b, name='add')

	return conv

def conv_block(x, out_depth, train_logical, w_initializer, b_initializer, scope):

	with tf.variable_scope(scope):

		conv=conv_layer(x, out_depth, [5,5], [2,2],w_initializer, b_initializer, 'conv')
		bn = batch_norm(conv, train_logical=train_logical, epsilon=1e-5, decay = 0.99, scope='bn')
		act = lrelu(bn, leak = 0.2, name='act')

	return act

def fc_block(input, in_num, out_num, w_initializer, b_initializer, scope):

	with tf.variable_scope(scope):

		w = tf.get_variable('w', [in_num, out_num], dtype=tf.float32, initializer=w_initializer)
		b = tf.get_variable('b', [out_num], dtype=tf.float32, initializer=b_initializer)
	
		fc = tf.matmul(input, w, name='matmul')
		fc = tf.add(fc, b, name='add')

	return fc	

def encoder(input, train_logical, latent_dim):

	xavier_initializer_conv = tf.contrib.layers.xavier_initializer_conv2d()
	xavier_initializer_fc = tf.contrib.layers.xavier_initializer()
	zeros_initializer = tf.zeros_initializer()

	act1 = conv_block(input, D1, train_logical, xavier_initializer_conv, zeros_initializer, 'conv1')
	act2 = conv_block(act1, D2, train_logical, xavier_initializer_conv, zeros_initializer, 'conv2')
	act3 = conv_block(act2, D3, train_logical, xavier_initializer_conv, zeros_initializer, 'conv3')
	act4 = conv_block(act3, D4, train_logical, xavier_initializer_conv, zeros_initializer, 'conv4')

	act4_num = int(np.prod(act4.get_shape()[1:]))
	act4_flat = tf.reshape(act4, [-1, act4_num])

	mean = fc_block(act4_flat, act4_num, latent_dim, xavier_initializer_fc, zeros_initializer, 'mean')
	stddev = fc_block(act4_flat, act4_num, latent_dim, xavier_initializer_fc, zeros_initializer, 'stddev')

	return mean, stddev

# -------------------------------------------------------------
# -------------------------- decoder --------------------------
# -------------------------------------------------------------

def conv_transpose_layer(x, out_depth, kernel, strides, w_initializer, b_initializer,scope):

	with tf.name_scope(scope): 

		in_shape = x.get_shape().as_list()
		in_batch = tf.shape(x)[0]
		in_height, in_width, in_depth = in_shape[1:]
		out_shape = [in_batch, in_height*strides[0],in_width*strides[1],out_depth]

		w = tf.get_variable('w', shape=[kernel[0], kernel[1], out_depth, in_depth], initializer= w_initializer)
		b = tf.get_variable('b', shape=[out_depth], initializer=b_initializer)

		conv = tf.nn.conv2d_transpose(x, filter=w, output_shape=out_shape, strides=[1, strides[0], strides[1], 1], padding='SAME', name='deconv')
		conv = tf.add(conv, b, name='add')

	return conv	

def decoder_conv_block(x, depth, train_logical, w_initializer, b_initializer, scope, final=False):

	with tf.variable_scope(scope): 

		conv = conv_transpose_layer(x, depth, kernel=[5,5], strides=[2,2], w_initializer=w_initializer, b_initializer=b_initializer,scope = 'conv')

		if final:
			act = tf.nn.sigmoid(conv, name='act')
		else:
			bn = batch_norm(conv, train_logical=train_logical, epsilon=1e-5, decay=0.99, scope='bn')
			act = tf.nn.relu(bn, name='act')

	return act

def decoder_fc_block(x, height, width, depth, train_logical, w_initializer, b_initializer, scope):

	latent_dim = x.get_shape()[1]

	with tf.variable_scope(scope):
	
		w = tf.get_variable('w', shape=[latent_dim, height * width * depth], dtype=tf.float32, initializer=w_initializer)
		b = tf.get_variable('b', shape=[height * width * depth], dtype=tf.float32, initializer=b_initializer)
		flat_conv = tf.add(tf.matmul(x, w), b, name='flat_conv')

		conv = tf.reshape(flat_conv, shape=[-1, height, width, depth], name='conv')
		bn = batch_norm(conv, train_logical=train_logical, epsilon=1e-5, decay=0.99, scope='bn')
		act = tf.nn.relu(bn, name='act')

	return act

def decoder(input, train_logical):

	xavier_initializer_conv = tf.contrib.layers.xavier_initializer_conv2d()
	xavier_initializer_fc = tf.contrib.layers.xavier_initializer()
	zeros_initializer = tf.zeros_initializer()

	act1 = decoder_fc_block(input,H1,W1,D4,train_logical,xavier_initializer_fc,zeros_initializer, 'fc1')

	act2 = decoder_conv_block(act1, D3, train_logical,xavier_initializer_conv,zeros_initializer, 'conv2')
	act3 = decoder_conv_block(act2, D2, train_logical,xavier_initializer_conv,zeros_initializer, 'conv3')
	act4 = decoder_conv_block(act3, D1, train_logical,xavier_initializer_conv,zeros_initializer, 'conv4')
	act5 = decoder_conv_block(act4, DEPTH, train_logical,xavier_initializer_conv,zeros_initializer, 'conv5', final=True)

	return act5

# -------------------------------------------------------------
# -------------------------- training --------------------------
# --------------------------------------------------------------

def make_dir(directory):
	
	if not os.path.exists(directory):
		os.makedirs(directory)

def train():

	with tf.device('/cpu:0'):
		with tf.name_scope('batch'):
			batch_image = read_record('./train.record', BATCH_SIZE, HEIGHT, WIDTH, DEPTH)

	with tf.variable_scope('encoder'):
		latent_mean, latent_stddev = encoder(batch_image, train_logical=True, latent_dim=LATENT_DIM)

	with tf.variable_scope('variance'):
		random_normal = tf.random_normal([BATCH_SIZE,LATENT_DIM], 0.0, 1.0, dtype=tf.float32)
		latent_vec = latent_mean + tf.multiply(random_normal, latent_stddev)
	
	latent_sample = tf.placeholder(tf.float32, shape=[None, LATENT_DIM], name='latent_input')
    
	with tf.variable_scope('decoder') as scope:
		with tf.name_scope('train'):
			y = decoder(latent_vec, train_logical=True)
		scope.reuse_variables()
		with tf.name_scope('eval'):
			gen_image = decoder(latent_sample, train_logical=False)

	with tf.name_scope('loss'):
		kl_divergence = -0.5*tf.reduce_sum(1 + 2*latent_stddev - tf.square(latent_mean) - tf.exp(2*latent_stddev))
		reconstruction_loss = tf.reduce_sum(tf.square(y - batch_image))

	with tf.name_scope('optimizer'):
		vae_loss = reconstruction_loss + kl_divergence
		train_step = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(vae_loss)

	sess = tf.Session()
	saver = tf.train.Saver()
	sess.run(tf.global_variables_initializer())
	coord = tf.train.Coordinator()
	threads = tf.train.start_queue_runners(sess=sess, coord=coord)
		
	for i in range(TRAIN_ITERS):

		_, loss =sess.run([train_step,vae_loss])

		print('iter: %d, loss:%f' % (i, loss))

		if (i+1)%SAVE_INTERVAL == 0:
			
			make_dir(MODEL_PATH)
			saver.save(sess, MODEL_PATH + '/vae', global_step=i+1)

		if (i+1)%EVAL_INTERVAL == 0:

			make_dir(SAMPLES_PATH)
			latent_random = np.random.normal(0.0, 1.0, size=[EVAL_ROWS*EVAL_COLS, LATENT_DIM]).astype(np.float32)
			data = sess.run(gen_image, feed_dict={latent_sample: latent_random})
			data = np.reshape(data*255,(EVAL_ROWS, EVAL_COLS, HEIGHT, WIDTH, DEPTH))
			data = np.concatenate(np.concatenate(data, 1), 1)          
			cv2.imwrite(SAMPLES_PATH + '/iter-' + str(i) + '.png', data)

	saver.save(sess, MODEL_PATH + '/gan', global_step=i+1)

# -------------------------------------------------------------------
# -------------------------- record reader --------------------------
# -------------------------------------------------------------------

def read_example(filename, height, width, depth):

	reader = tf.FixedLengthRecordReader(height*width*depth)
	filename_queue = tf.train.string_input_producer([filename], num_epochs=None)
	key, serialized_example = reader.read(filename_queue)

	image_raw = tf.decode_raw(serialized_example, tf.uint8)
	image = tf.cast(tf.reshape(image_raw, [height, width, depth]), tf.float32)

	return image

def make_batch(image, batch_size):

	min_queue_examples = 100

	batch_images = tf.train.shuffle_batch([image], batch_size=batch_size, 
					      capacity=min_queue_examples + 10*batch_size,
					      min_after_dequeue=min_queue_examples, num_threads=4)

	return batch_images

def read_record(filename, batch_size, height, width, depth):
	
	image = read_example(filename, height, width, depth)
	image = image/255.0
	batch_images = make_batch(image,batch_size)

	return batch_images

# -------------------------------------------------------------------------
# -------------------------- batch normalization --------------------------
# -------------------------------------------------------------------------

def assign_decay(orig_val, new_val, momentum, name):

	with tf.name_scope(name):

		scaled_diff = (1 - momentum) * (new_val - orig_val)

	return tf.assign_add(orig_val, scaled_diff)

def batch_norm(x, train_logical, decay, epsilon, scope=None, shift=True, scale=False):

	channels = x.get_shape()[-1]
	ndim = len(x.shape)

	with tf.variable_scope(scope):

		moving_m = tf.get_variable('mean', [channels], initializer=tf.zeros_initializer, trainable=False)
		moving_v = tf.get_variable('var', [channels], initializer=tf.ones_initializer, trainable=False)

		if train_logical == True:

			m, v = tf.nn.moments(x, range(ndim - 1))
			update_m = assign_decay(moving_m, m, decay, 'update_mean')
			update_v = assign_decay(moving_v, v, decay, 'update_var')

			with tf.control_dependencies([update_m, update_v]):
				output = (x - m) * tf.rsqrt(v + epsilon)

		else:
			m, v = moving_m, moving_v
			output = (x - m) * tf.rsqrt(v + epsilon)

		if scale:
			output *= tf.get_variable('gamma', [channels], initializer=tf.ones_initializer)

		if shift:
			output += tf.get_variable('beta', [channels], initializer=tf.zeros_initializer)

	return output

# -------------------------------------------------------------------------
# -------------------------------------------------------------------------
# -------------------------------------------------------------------------

if __name__ == "__main__":
    train()
