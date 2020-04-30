import os
import shutil
import sys
import time

import tensorflow.compat.v1 as tf
import numpy as np

from utils import Logger
from utils import DEFINE_boolean
from utils import DEFINE_float
from utils import DEFINE_integer
from utils import DEFINE_string
from utils import print_user_flags

import data_utils

from micro_controller import MicroController
from micro_child import MicroChild

flags = tf.app.flags
FLAGS = flags.FLAGS


################## YOU Should write under parameter ######################
DEFINE_string("output_dir", "./output" , "")
DEFINE_string("train_data_dir", "./data/train", "")
DEFINE_string("val_data_dir", "./data/valid", "")
DEFINE_string("test_data_dir", "./data/test", "")
DEFINE_integer("channel",1, "MNIST: 1, Cifar10: 3")
DEFINE_integer("img_size", 32, "enlarge image size")
DEFINE_integer("n_aug_img",1 , "if 2: num_img: 55000 -> aug_img: 110000, elif 1: False")
##########################################################################

DEFINE_boolean("reset_output_dir", True, "Delete output_dir if exists.")
DEFINE_string("data_format","NHWC", "'NHWC or NCHW'")
DEFINE_string("search_for", "micro","")

DEFINE_integer("batch_size",128,"")
DEFINE_integer("num_epochs", 100," = (10+ 20+ 40+ 80)") #original 150

DEFINE_integer("child_lr_dec_every", 100, "")
DEFINE_integer("child_num_layers", 6, "Number of layer. IN this case we will calculate 4 conv and 2 pooling layers") # default 6
DEFINE_integer("child_num_cells", 5, "child_num_cells +2 = Number of DAG'S Nodes") #default 5
DEFINE_integer("child_filter_size", 5, "")
DEFINE_integer("child_out_filters", 20, "")
DEFINE_integer("child_out_filters_scale", 1, "")
DEFINE_integer("child_num_branches", 5, "It should be same with number of kernel operation to calculate.")
DEFINE_integer("child_num_aggregate", None, "")
DEFINE_integer("child_num_replicas", 1, "")
DEFINE_integer("child_block_size", 3, "")
DEFINE_integer("child_lr_T_0", 10, "for lr schedule")
DEFINE_integer("child_lr_T_mul", 2, "for lr schedule")
DEFINE_integer("child_cutout_size", None, "CutOut size")
DEFINE_float("child_grad_bound", 5.0, "Gradient clipping")
DEFINE_float("child_lr", 0.1, "")
DEFINE_float("child_lr_dec_rate", 0.1, "")
DEFINE_float("child_keep_prob", 0.9, "")
DEFINE_float("child_drop_path_keep_prob", 0.6, "minimum drop_path_keep_prob")
DEFINE_float("child_l2_reg", 1e-4, "")
DEFINE_float("child_lr_max", 0.05, "for lr schedule")
DEFINE_float("child_lr_min", 0.0005, "for lr schedule")
DEFINE_string("child_skip_pattern", None, "Must be ['dense', None]")
DEFINE_string("child_fixed_arc", None, "")
DEFINE_boolean("child_use_aux_heads", True, "Should we use an aux head")
DEFINE_boolean("child_sync_replicas", False, "To sync or not to sync.")
DEFINE_boolean("child_lr_cosine", True, "Use cosine lr schedule")

DEFINE_float("controller_lr", 0.0035, "")
DEFINE_float("controller_lr_dec_rate", 1.0, "")
DEFINE_float("controller_keep_prob", 0.5, "")
DEFINE_float("controller_l2_reg", 0.0, "")
DEFINE_float("controller_bl_dec", 0.99, "")
DEFINE_float("controller_tanh_constant", 1.10, "")
DEFINE_float("controller_op_tanh_reduce", 2.5, "")
DEFINE_float("controller_temperature", None, "")
DEFINE_float("controller_entropy_weight", 0.0001, "")
DEFINE_float("controller_skip_target", 0.8, "")
DEFINE_float("controller_skip_weight", 0.0, "")
DEFINE_integer("controller_num_aggregate", 10, "")
DEFINE_integer("controller_num_replicas", 1, "")
DEFINE_integer("controller_train_steps", 30, "")
DEFINE_integer("controller_forwards_limit", 2, "")
DEFINE_integer("controller_train_every", 1,
			   "train the controller after this number of epochs")
DEFINE_boolean("controller_search_whole_channels", True, "")
DEFINE_boolean("controller_sync_replicas", True, "To sync or not to sync.")
DEFINE_boolean("controller_training", True, "")
DEFINE_boolean("controller_use_critic", False, "")

DEFINE_integer("log_every", 50, "How many steps to log")
DEFINE_integer("eval_every_epochs", 1, "How many epochs to eval")

channel = FLAGS.channel

def get_ops(images, labels):
	"""
	Args:
	  images: dict with keys {"train", "valid", "test"}.
	  labels: dict with keys {"train", "valid", "test"}.
	"""

	ControllerClass = MicroController
	ChildClass = MicroChild

	child_model = ChildClass(
		images,
		labels,
		use_aux_heads=FLAGS.child_use_aux_heads,
		cutout_size=FLAGS.child_cutout_size,
		whole_channels=FLAGS.controller_search_whole_channels,
		num_layers=FLAGS.child_num_layers,
		num_cells=FLAGS.child_num_cells,
		num_branches=FLAGS.child_num_branches,
		fixed_arc=FLAGS.child_fixed_arc,
		out_filters_scale=FLAGS.child_out_filters_scale,
		out_filters=FLAGS.child_out_filters,
		keep_prob=FLAGS.child_keep_prob,
		drop_path_keep_prob=FLAGS.child_drop_path_keep_prob,
		num_epochs=FLAGS.num_epochs,
		l2_reg=FLAGS.child_l2_reg,
		data_format=FLAGS.data_format,
		batch_size=FLAGS.batch_size,
		clip_mode="norm",
		grad_bound=FLAGS.child_grad_bound,
		lr_init=FLAGS.child_lr,
		lr_dec_every=FLAGS.child_lr_dec_every,
		lr_dec_rate=FLAGS.child_lr_dec_rate,
		lr_cosine=FLAGS.child_lr_cosine,
		lr_max=FLAGS.child_lr_max,
		lr_min=FLAGS.child_lr_min,
		lr_T_0=FLAGS.child_lr_T_0,
		lr_T_mul=FLAGS.child_lr_T_mul,
		optim_algo="momentum",
		sync_replicas=FLAGS.child_sync_replicas,
		num_aggregate=FLAGS.child_num_aggregate,
		num_replicas=FLAGS.child_num_replicas,
		channel=FLAGS.channel)

	if FLAGS.child_fixed_arc is None:
		controller_model = ControllerClass(
			search_for=FLAGS.search_for,
			search_whole_channels=FLAGS.controller_search_whole_channels,
			skip_target=FLAGS.controller_skip_target,
			skip_weight=FLAGS.controller_skip_weight,
			num_cells=FLAGS.child_num_cells,
			num_layers=FLAGS.child_num_layers,
			num_branches=FLAGS.child_num_branches,
			out_filters=FLAGS.child_out_filters,
			lstm_size=64,
			lstm_num_layers=1,
			lstm_keep_prob=1.0,
			tanh_constant=FLAGS.controller_tanh_constant,
			op_tanh_reduce=FLAGS.controller_op_tanh_reduce,
			temperature=FLAGS.controller_temperature,
			lr_init=FLAGS.controller_lr,
			lr_dec_start=0,
			lr_dec_every=1000000,  # never decrease learning rate
			l2_reg=FLAGS.controller_l2_reg,
			entropy_weight=FLAGS.controller_entropy_weight,
			bl_dec=FLAGS.controller_bl_dec,
			use_critic=FLAGS.controller_use_critic,
			optim_algo="adam",
			sync_replicas=FLAGS.controller_sync_replicas,
			num_aggregate=FLAGS.controller_num_aggregate,
			num_replicas=FLAGS.controller_num_replicas)

		child_model.connect_controller(controller_model)
		controller_model.build_trainer(child_model)

		controller_ops = {
			"train_step": controller_model.train_step,
			"loss": controller_model.loss,
			"train_op": controller_model.train_op,
			"lr": controller_model.lr,
			"grad_norm": controller_model.grad_norm,
			"valid_acc": controller_model.valid_acc,
			"optimizer": controller_model.optimizer,
			"baseline": controller_model.baseline,
			"entropy": controller_model.sample_entropy,
			"sample_arc": controller_model.sample_arc,
			"skip_rate": controller_model.skip_rate,
			"normal_arc": controller_model.current_normal_arc,
			"reduce_arc": controller_model.current_reduce_arc,
		}

	else:
		assert not FLAGS.controller_training, (
			"--child_fixed_arc is given, cannot train controller")
		child_model.connect_controller(None)
		controller_ops = None

	child_ops = {
		"global_step": child_model.global_step,
		"loss": child_model.loss,
		"train_op": child_model.train_op,
		"lr": child_model.lr,
		"grad_norm": child_model.grad_norm,
		"train_acc": child_model.train_acc,
		"optimizer": child_model.optimizer,
		"num_train_batches": child_model.num_train_batches,
		"normal_arc": child_model.current_normal_arc,
		"reduce_arc": child_model.current_reduce_arc,
	}

	ops = {
		"child": child_ops,
		"controller": controller_ops,
		"eval_every": child_model.num_train_batches * FLAGS.eval_every_epochs,
		"eval_func": child_model.eval_once,
		"num_train_batches": child_model.num_train_batches,
	}

	return ops



def train():
	
	def _split_arc_seq(arc_seq):
		arc_seq_length = len(arc_seq)
		arc_nodes_amt = FLAGS.child_num_cells
		assert arc_seq_length%arc_nodes_amt == 0
		arc_nodes = np.split(arc_seq, arc_nodes_amt)
		return arc_nodes
	
	normal_train_dict = {}
	reduce_train_dict = {}
	
	def _save_trained_op(key, arc_type)
		if arc_type is "normal":
			if key in normal_train_dict:
				normal_train_dict[key]+=1
			else
				normal_train_dict[key]=1
		elif arc_type is "reduce":
			if key in reduce_train_dict:
				reduce_train_dict[key]+=1
			else
				reduce_train_dict[key]=1
	
	def _save_trained_arc(arc_seq, arc_type):
		arc_nodes = _split_arc_seq(arc_seq)
		for i, node in enumerate(arc_nodes):
			x_op = node[1]
			y_op = node[3]
			x_key = "node"+str(i)+"_op"+str(x_op)
			y_key = "node"+str(i)+"_op"+str(y_op)
			_save_trained_op(x_key, arc_type)
			_save_trained_op(y_key, arc_type)
		
	def _get_trained_op(key, arc_type)
		if arc_type is "normal":
			if key in normal_train_dict:
				return normal_train_dict[key]
			else
				return 0
		elif arc_type is "reduce":
			if key in reduce_train_dict:
				return reduce_train_dict[key]
			else
				return 0
			
	def _get_trained_arc(arc_seq, arc_type):
		arc_nodes = _split_arc_seq(arc_seq)
		trained_arc = []
		for i, node in enumerate(arc_nodes):
			x_op = node[1]
			y_op = node[3]
			x_key = "node"+str(i)+"_op"+str(x_op)
			y_key = "node"+str(i)+"_op"+str(y_op)
			x_train_amt = _get_trained_op(x_key, arc_type)
			y_train_amt = _get_trained_op(y_key, arc_type)
			trained_arc.append(x_train_amt)
			trained_arc.append(y_train_amt)
		
		return trained_arc
		
	
	images, labels = data_utils.read_data(FLAGS.train_data_dir,
										  FLAGS.val_data_dir,
										  FLAGS.test_data_dir,
										  FLAGS.channel,
										  FLAGS.img_size,
										  FLAGS.n_aug_img)
	
	n_data = np.shape(images["train"])[0]
	print("Number of training data: %d" % (n_data))
	

	g = tf.Graph()
	with g.as_default():
		ops =get_ops(images, labels)
		child_ops = ops["child"]
		controller_ops = ops["controller"]

		saver = tf.train.Saver(max_to_keep=2)
		checkpoint_saver_hook = tf.train.CheckpointSaverHook(
			FLAGS.output_dir, save_steps=child_ops["num_train_batches"], saver=saver)
		hooks = []
		#hooks = [checkpoint_saver_hook]
		if FLAGS.child_sync_replicas:
			sync_replicas_hook = child_ops["optimizer"].make_session_run_hook(True)
			hooks.append(sync_replicas_hook)
		if FLAGS.controller_training and FLAGS.controller_sync_replicas:
			sync_replicas_hook = controller_ops["optimizer"].make_session_run_hook(True)
			hooks.append(sync_replicas_hook)

		print("-" * 80)
		print("Starting session")
		config = tf.ConfigProto(allow_soft_placement=True)
		with tf.train.SingularMonitoredSession(
				config=config, hooks=hooks, checkpoint_dir=FLAGS.output_dir) as sess:
			start_time = time.time()
			while True:
				run_ops = [
					child_ops["loss"],
					child_ops["lr"],
					child_ops["grad_norm"],
					child_ops["train_acc"],
					child_ops["train_op"],
					child_ops["normal_arc"],
					child_ops["reduce_arc"]]
				loss, lr, gn, tr_acc, _, normal_arc, reduce_arc = sess.run(run_ops)
				global_step = sess.run(child_ops["global_step"])
				_save_trained_arc(normal_arc, "normal")
				_save_trained_arc(reduce_arc, "reduce")
				normal_train_amt = _get_trained_arc(normal_arc, "normal")
				reduce_train_amt = _get_trained_arc(reduce_arc, "reduce")

				if FLAGS.child_sync_replicas:
					actual_step = global_step * FLAGS.num_aggregate
				else:
					actual_step = global_step
				epoch = actual_step // ops["num_train_batches"]
				curr_time = time.time()
				if global_step % FLAGS.log_every == 0:
					log_string = ""
					log_string += "epoch = {:<6d}".format(epoch)
					log_string += "ch_step = {:<6d}".format(global_step)
					log_string += " loss = {:<8.6f}".format(loss)
					log_string += "   lr = {:<8.4f}".format(lr)
					log_string += " |g| = {:<8.4f}".format(gn)
					log_string += " tr_acc = {:<3d}/{:>3d}".format(
						tr_acc, FLAGS.batch_size)
					log_string += "   mins = {:<10.2f}".format(
						float(curr_time - start_time) / 60)
					print(log_string)
					print("\tNormal architecture: \n\t",normal_arc)
					print("\tTrain amount: \n\t",normal_arc)
					print("\tReduce architecture: \n\t",reduce_arc)
					print("\tTrain amount: \n\t",reduce_arc)

				if actual_step % ops["eval_every"] == 0:
					if (FLAGS.controller_training and
							epoch % FLAGS.controller_train_every == 0):
						print("Epoch {}: Training controller".format(epoch))
						for ct_step in range(FLAGS.controller_train_steps *
											 FLAGS.controller_num_aggregate):
							run_ops = [
								controller_ops["loss"],
								controller_ops["entropy"],
								controller_ops["lr"],
								controller_ops["grad_norm"],
								controller_ops["valid_acc"],
								controller_ops["baseline"],
								controller_ops["skip_rate"],
								controller_ops["train_op"],
								controller_ops["normal_arc"],
								controller_ops["reduce_arc"],
							]
							loss, entropy, lr, gn, val_acc, bl, skip, _, normal_arc, reduce_arc = sess.run(run_ops)
							controller_step = sess.run(controller_ops["train_step"])

							if ct_step % FLAGS.log_every == 0:
								curr_time = time.time()
								log_string = ""
								log_string += "ctrl_step = {:<6d}".format(controller_step)
								log_string += " loss = {:<7.3f}".format(loss)
								log_string += " ent = {:<5.2f}".format(entropy)
								log_string += "   lr = {:<6.4f}".format(lr)
								log_string += "   |g| = {:<8.4f}".format(gn)
								log_string += " acc = {:<6.4f}".format(val_acc)
								log_string += "   bl = {:<5.2f}".format(bl)
								log_string += "  mins = {:<.2f}".format(
									float(curr_time - start_time) / 60)
								print(log_string)
								print("\tNormal architecture: \n\t",normal_arc)
								print("\tReduce architecture: \n\t",reduce_arc)

						print("Here are 10 architectures")
						for _ in range(10):
							arc, acc = sess.run([
								controller_ops["sample_arc"],
								controller_ops["valid_acc"],
							])
							if FLAGS.search_for == "micro":
								normal_arc, reduce_arc = arc
								print(np.reshape(normal_arc, [-1]))
								print(np.reshape(reduce_arc, [-1]))
							else:
								start = 0
								for layer_id in range(FLAGS.child_num_layers):
									if FLAGS.controller_search_whole_channels:
										end = start + 1 + layer_id
									else:
										end = start + 2 * FLAGS.child_num_branches + layer_id
									print(np.reshape(arc[start: end], [-1]))
									start = end
							print("val_acc = {:<6.4f}".format(acc))
							print("-" * 80)

					print("Epoch {}: Eval".format(epoch))
					if FLAGS.child_fixed_arc is None:
						ops["eval_func"](sess, "valid")
					ops["eval_func"](sess, "test")

				if epoch >= FLAGS.num_epochs:
					break

def main(_):
	print("-" * 80)
	if not os.path.isdir(FLAGS.output_dir):
		print("Path {} does not exist. Creating.".format(FLAGS.output_dir))
		os.makedirs(FLAGS.output_dir)
	elif FLAGS.reset_output_dir:
		print("Path {} exists. Remove and remake.".format(FLAGS.output_dir))
		shutil.rmtree(FLAGS.output_dir)
		os.makedirs(FLAGS.output_dir)

	print("-" * 80)
	log_file = os.path.join(FLAGS.output_dir, "stdout")
	print("Logging to {}".format(log_file))
	sys.stdout = Logger(log_file)
	print_user_flags()
	train()

if __name__ == "__main__":
	tf.app.run()
