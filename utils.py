import sys
import numpy as np
import tensorflow.compat.v1 as tf
import matplotlib.pyplot as plt
import os, errno

user_flags = []

def DEFINE_string(name, default_value, doc_string):
	tf.app.flags.DEFINE_string(name, default_value, doc_string)
	global user_flags
	user_flags.append(name)

def DEFINE_integer(name, default_value, doc_string):
	tf.app.flags.DEFINE_integer(name, default_value, doc_string)
	global user_flags
	user_flags.append(name)

def DEFINE_float(name, defualt_value, doc_string):
	tf.app.flags.DEFINE_float(name, defualt_value, doc_string)
	global user_flags
	user_flags.append(name)

def DEFINE_boolean(name, default_value, doc_string):
	tf.app.flags.DEFINE_boolean(name, default_value, doc_string)
	global user_flags
	user_flags.append(name)
	
def silently_remove_file(filename):
	try:
		os.remove(filename)
	except OSError as e: # this would be "except OSError, e:" before Python 2.6
		if e.errno != errno.ENOENT: # errno.ENOENT = no such file or directory
			raise # re-raise exception if a different error occurred

def print_user_flags(line_limit = 80):
	print("-" * 80)

	global user_flags
	FLAGS = tf.app.flags.FLAGS

	for flag_name in sorted(user_flags):
		value = "{}".format(getattr(FLAGS, flag_name))
		log_string = flag_name
		log_string += "." * (line_limit - len(flag_name) - len(value))
		log_string += value
		print(log_string)
		
def plot_data_label(images, labels, channels, width, height, figsize):
	# width = the number of images placed horizontally
	# height = the number of image placed vertically
	# figsize = white space between images
	order = 1
	load_size = images.shape[1]
	fig, axes = plt.subplots(width, height, figsize=(figsize, figsize))
	fig.subplots_adjust(hspace=1, wspace=1)
	path = os.getcwd() + "/plt_images"
	if not os.path.exists(path):
		os.makedirs(path)
	for i, ax in enumerate(axes.flat):
		if channels == 1:
			ax.imshow(images[i].reshape(load_size, load_size))
		else:
			ax.imshow(images[i].reshape(load_size, load_size, channels))
		ax.set_xlabel("label: %d" % (labels[i]))
		ax.set_xticks([])
		ax.set_yticks([])
	file_name = path + "/image" + str(order) + ".png"
	while os.path.isfile(file_name) is True:
		order += 1
		file_name = path + "/image" + str(order) + ".png"
	plt.savefig(file_name)
	plt.close()        
	
class StackStructure():
	def __init__(self, stack_size = 10):
		self.stack_size = stack_size
		self.storage = [[] for i in range(self.stack_size)]
	def push(self, data):
		# remove oldest
		self.storage.pop(0)
		# add newest
		self.storage.append(data)
	def get_newest(self):
		return self.storage[-1]
	def get_oldest(self):
		return self.storage[0]
	def get_full_stack(self, reverse=False):
		if reverse:
			return self.storage[::-1]
		else:
			return self.storage
		
class ArcOrderedList():
	def __init__(self, list_size = 20):
		self.list_size = list_size
		self.storage = [{"normal_arc":[],"reduce_arc":[],"acc":0.0,"added_at_epoch":-1} for i in range(self.list_size)]
	
	def add_arc(self, normal_arc, reduce_arc, acc, current_epoch):
		found = False
		i = 0
		temp_arc = {
						"normal_arc":normal_arc,
						"reduce_arc":reduce_arc,
						"acc":acc,
						"added_at_epoch":current_epoch
					}
		
		while not found:
			if i == self.list_size:
				self.storage.pop(0)
				self.storage.append(temp_arc)
				found = True
			elif self.storage[i]["acc"] > acc:
				self.storage.insert(i,temp_arc)
				self.storage.pop(0)
				found = True
			else:
				i+=1
		return i
	
	def get_best_arc(self):
		return self.storage[-1]
	
	def get_best_acc(self):
		return self.get_best_arc()["acc"]
	
	def get_last_best_epoch(self):
		return self.get_best_arc()["added_at_epoch"]
	
	def get_list_as_csv_data(self):
		csv_string = ""
		for i, arc in enumerate(self.storage):
			temp_string = ""
			## normal arc
			temp_string += "["
			temp_string += ','.join(str(x) for x in arc["normal_arc"])
			temp_string += "];"
			## reduce arc
			temp_string += "["
			temp_string += ','.join(str(x) for x in arc["reduce_arc"])
			temp_string += "];"
			## acc
			temp_string += str(arc["acc"])
			temp_string += ";"
			## added_at_epoch
			temp_string += str(arc["added_at_epoch"])
			
			
			temp_string += "\n"
			
			csv_string += temp_string
		
		return csv_string
	


class MovingAverageStructure():
	
	def __init__(self, window_size, dtype=np.int32):
		self.window_size = window_size
		self.dtype = dtype
		self.storage = np.zeros(window_size, dtype=self.dtype)
	
	def push(self, data):
		self.storage = np.roll(self.storage, 1)
		self.storage[0] = data
	
	def get_mov_average(self):
		indices = np.nonzero(self.storage)[0]
		count = 0.0
		if len(indices) == 0:
			return count
		for i in indices:
			count += self.storage[i]
		n = len(indices)
		return float(count)/float(n)
	
	


class Logger(object):
	def __init__(self, output_file):
		self.terminal = sys.stdout
		self.log = open(output_file, "a")

	def flush(self):
		pass

	def write(self, message):
		self.terminal.write(message)
		self.terminal.flush()
		self.log.write(message)
		self.log.flush()

def make_one_hot(list):
	n_values = np.max(list) + 1
	one_hot = np.eye(n_values)[list]
	return one_hot


def count_model_params(tf_variables):
	"""
	Args:
	  tf_variables: list of all model variables
	"""

	num_vars = 0
	for var in tf_variables:
		num_vars += np.prod([dim for dim in var.get_shape()])
	return num_vars


def get_train_ops(
		loss,
		tf_variables,
		train_step,
		clip_mode=None,
		grad_bound=None,
		l2_reg=1e-4,
		lr_warmup_val=None,
		lr_warmup_steps=100,
		lr_init=0.1,
		lr_dec_start=0,
		lr_dec_every=10000,
		lr_dec_rate=0.1,
		lr_dec_min=None,
		lr_cosine=False,
		lr_max=None,
		lr_min=None,
		lr_T_0=None,
		lr_T_mul=None,
		num_train_batches=None,
		optim_algo=None,
		sync_replicas=False,
		num_aggregate=None,
		num_replicas=None,
		get_grad_norms=False,
		moving_average=None):
	"""
	Args:
	  clip_mode: "global", "norm", or None.
	  moving_average: store the moving average of parameters
	"""

	if l2_reg > 0:
		l2_losses = []
		for var in tf_variables:
			l2_losses.append(tf.reduce_sum(var ** 2))
		l2_loss = tf.add_n(l2_losses)
		loss += l2_reg * l2_loss  # loss = loss + 1e-4*l2_loss

	grads = tf.gradients(loss, tf_variables)
	grad_norm = tf.global_norm(grads)

	grad_norms = {}
	for v, g in zip(tf_variables, grads):
		if v is None or g is None:
			continue
		if isinstance(g, tf.IndexedSlices):
			grad_norms[v.name] = tf.sqrt(tf.reduce_sum(g.values ** 2))
		else:
			grad_norms[v.name] = tf.sqrt(tf.reduce_sum(g ** 2))

	if clip_mode is not None:
		assert grad_bound is not None, "Need grad_bound to clip gradients."
		if clip_mode == "global":
			grads, _ = tf.clip_by_global_norm(grads, grad_bound)
		elif clip_mode == "norm":
			clipped = []
			for g in grads:
				if isinstance(g, tf.IndexedSlices):
					c_g = tf.clip_by_norm(g.values, grad_bound)
					c_g = tf.IndexedSlices(g.indices, c_g)
				else:
					c_g = tf.clip_by_norm(g, grad_bound)
				clipped.append(g)
			grads = clipped
		else:
			raise NotImplementedError("Unknown clip_mode {}".format(clip_mode))

	if lr_cosine:
		assert lr_max is not None, "Need lr_max to use lr_cosine"
		assert lr_min is not None, "Need lr_min to use lr_cosine"
		assert lr_T_0 is not None, "Need lr_T_0 to use lr_cosine"
		assert lr_T_mul is not None, "Need lr_T_mul to use lr_cosine"
		assert num_train_batches is not None, ("Need num_train_batches to use"
											   " lr_cosine")

		curr_epoch = train_step // num_train_batches  # train step will be calculated by just one batch!

		last_reset = tf.Variable(0, dtype=tf.int32, trainable=False,
								 name="last_reset")
		T_i = tf.Variable(lr_T_0, dtype=tf.int32, trainable=False, name="T_i")
		T_curr = curr_epoch - last_reset

		def _update():
			update_last_reset = tf.assign(last_reset, curr_epoch, use_locking=True)
			update_T_i = tf.assign(T_i, T_i * lr_T_mul, use_locking=True)
			with tf.control_dependencies([update_last_reset, update_T_i]):
				rate = tf.to_float(T_curr) / tf.to_float(T_i) * 3.1415926
				lr = lr_min + 0.5 * (lr_max - lr_min) * (1.0 + tf.cos(rate))
			return lr

		def _no_update():
			rate = tf.to_float(T_curr) / tf.to_float(T_i) * 3.1415926
			lr = lr_min + 0.5 * (lr_max - lr_min) * (1.0 + tf.cos(rate))
			return lr

		learning_rate = tf.cond(
			tf.greater_equal(T_curr, T_i), _update, _no_update)
	else:
		learning_rate = tf.train.exponential_decay(
			lr_init, tf.maximum(train_step - lr_dec_start, 0), lr_dec_every,
			lr_dec_rate, staircase=True)
		if lr_dec_min is not None:
			learning_rate = tf.maximum(learning_rate, lr_dec_min)

	if lr_warmup_val is not None:
		learning_rate = tf.cond(tf.less(train_step, lr_warmup_steps),
								lambda: lr_warmup_val, lambda: learning_rate)

	if optim_algo == "momentum":
		opt = tf.train.MomentumOptimizer(
			learning_rate, 0.9, use_locking=True, use_nesterov=True)
	elif optim_algo == "sgd":
		opt = tf.train.GradientDescentOptimizer(learning_rate, use_locking=True)
	elif optim_algo == "adam":
		opt = tf.train.AdamOptimizer(learning_rate, beta1=0.0, epsilon=1e-3,
									 use_locking=True)
	else:
		raise ValueError("Unknown optim_algo {}".format(optim_algo))

	if sync_replicas:
		assert num_aggregate is not None, "Need num_aggregate to sync."
		assert num_replicas is not None, "Need num_replicas to sync."

		opt = tf.train.SyncReplicasOptimizer(
			opt,
			replicas_to_aggregate=num_aggregate,
			total_num_replicas=num_replicas,
			use_locking=True)

	if moving_average is not None:
		opt = tf.contrib.opt.MovingAverageOptimizer(
			opt, average_decay=moving_average)

	train_op = opt.apply_gradients(
		zip(grads, tf_variables), global_step=train_step)

	if get_grad_norms:
		return train_op, learning_rate, grad_norm, opt, grad_norms
	else:
		return train_op, learning_rate, grad_norm, opt
