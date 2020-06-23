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
from utils import silently_remove_file
from utils import MovingAverageStructure

import data_utils

from micro_controller import MicroController
from micro_child import MicroChild

import accuracy_scaling as accs

accuracy_scaling = accs.ArchitectureTrainingStorage()
flags = tf.app.flags
FLAGS = flags.FLAGS


################## YOU Should write under parameter ######################
DEFINE_string("output_dir", "./output" , "")
DEFINE_string("child_log_filename","child_log.txt","")
DEFINE_string("controller_log_filename","controller_log.txt","")
DEFINE_string("train_data_dir", "./data/train", "")
DEFINE_string("val_data_dir", "./data/valid", "")
DEFINE_string("test_data_dir", "./data/test", "")
DEFINE_integer("channel",2, "MNIST: 1, Cifar10: 3, parents: 3, parents_img: 2")
DEFINE_integer("img_size", 64, "enlarge image size")
DEFINE_integer("n_aug_img",1 , "if 2: num_img: 55000 -> aug_img: 110000, elif 1: False")
DEFINE_string("data_source","parents_img","either 'parents', 'mnist', 'cifar10', 'parents_img' ")
##########################################################################

DEFINE_boolean("reset_output_dir", True, "Delete output_dir if exists.")
DEFINE_string("data_format","NHWC", "'NHWC or NCHW'")
DEFINE_string("search_for", "micro","")

DEFINE_integer("batch_size",64,"") #original 128
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
DEFINE_float("child_lr_min", 0.001, "for lr schedule") #0.0005 original
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
DEFINE_float("controller_tanh_constant", 2.5, "") #original 1.10
DEFINE_float("controller_op_tanh_reduce", 2.5, "")
DEFINE_float("controller_temperature", 5.0, "") #original None
DEFINE_float("controller_entropy_weight", 0.0001, "")
DEFINE_float("controller_skip_target", 0.4, "") #original 0.8
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

DEFINE_integer("log_every", 5, "How many steps (how many batches) to log") # original 50
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
			"scaled_accuracy": controller_model.scaled_acc,
			"normal_arc_training": controller_model.normal_arc_training,
			"reduce_arc_training": controller_model.reduce_arc_training,
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
		"controller_model": controller_model
	}

	return ops



def train():
	images = {}
	labels = {}
	if FLAGS.data_source is "parents":
		images, labels = data_utils.parents_get_data(pathTrain = "parents_data/TrainSet.txt", pathTest="parents_data/TestSet.txt")
	elif FLAGS.data_source is "parents_img":
		images, labels = data_utils._parents_read_images()
	elif FLAGS.data_source is "cifar10":
		images, labels = data_utils._cifar10_load_data()
	else:
		images, labels = data_utils.read_data(FLAGS.train_data_dir,
											  FLAGS.val_data_dir,
											  FLAGS.test_data_dir,
											  FLAGS.channel,
											  FLAGS.img_size,
											  FLAGS.n_aug_img)
		
	n_data = np.shape(images["train"])[0]
	print("Number of training data: %d" % (n_data))
	

	g = tf.Graph()
	## creating log file names and removing old files if present
	child_log_filename = FLAGS.output_dir+"/"+FLAGS.child_log_filename
	silently_remove_file(child_log_filename)
	controller_log_filename = FLAGS.output_dir+"/"+FLAGS.controller_log_filename
	silently_remove_file(controller_log_filename)
	child_logfile = open(child_log_filename, "a+")
	controller_logfile = open(controller_log_filename, "a+")
	##
	
	## creating moving averages
	mov_avg_accuracy_struct = MovingAverageStructure(10,np.float32)
	mov_avg_training_struct = MovingAverageStructure(10,np.int32)
	with g.as_default():
		ops =get_ops(images, labels)
		
		print("images - train shape:", np.shape(images["train"]))
		print("labels - train shape:", np.shape(labels["train"]))
		print("images - valid shape:", np.shape(images["valid"]))
		print("labels - valid shape:", np.shape(labels["valid"]))
		print("images - test shape:", np.shape(images["test"]))
		print("labels - test shape:", np.shape(labels["test"]))
		
		#controller_model = ops["controller_model"]
		
		
		
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
				normal_train_amt = 0
				reduce_train_amt = 0
				normal_train_dict_length = 0
				reduce_train_dict_length = 0
				test_acc_scaling = 0
				
				
				accuracy_scaling.save_trained_arc(normal_arc, "normal")
				accuracy_scaling.save_trained_arc(reduce_arc, "reduce")
				normal_train_amt = accuracy_scaling.get_trained_arc(normal_arc, "normal")
				reduce_train_amt = accuracy_scaling.get_trained_arc(reduce_arc, "reduce")
				normal_train_dict_length = len(accuracy_scaling.normal_train_dict)
				reduce_train_dict_length = len(accuracy_scaling.reduce_train_dict)
					#test_acc_scaling = controller_model.accuracy_scaling.get_scaled_accuracy(0.5, normal_arc, reduce_arc, scaling_method="linear", arc_handling="sum")
				
				if FLAGS.child_sync_replicas:
					actual_step = global_step * FLAGS.num_aggregate
				else:
					actual_step = global_step
				epoch = actual_step // ops["num_train_batches"]
				curr_time = time.time()
				
				# LOGGING CHILD STEP
				## building log line
				### contains: [epoch];[global_step];[normal_arc];[reduce_arc];[elapsed_time];
				normal_arc_str = ','.join(['%d' % num for num in normal_arc])
				reduce_arc_str = ','.join(['%d' % num for num in reduce_arc])
				logline = str(epoch)+";"
				logline +=str(global_step)+";"
				logline +=normal_arc_str+";"
				logline +=reduce_arc_str+";"
				logline +="{:10.4f}".format(float(curr_time - start_time))
				
				child_logfile.write(logline+"\n")
				
				
				if global_step % FLAGS.log_every == 0:
					log_string = ""
					log_string += "epoch = {:<6d}".format(epoch)
					log_string += "ch_step = {:<6d}".format(global_step)
					log_string += " loss = {:<8.6f}".format(loss)
					log_string += "   lr = {:<8.4f}".format(lr)
					log_string += " |g| = {:<8.4f}".format(gn)
					log_string += " tr_acc = {:<3d}/{:>3d}".format(
						tr_acc, FLAGS.batch_size)
					log_string += " - {:<8.6f}".format(tr_acc/FLAGS.batch_size)
					log_string += "   mins = {:<10.2f}".format(
						float(curr_time - start_time) / 60)
					print(log_string)
					print("\tNormal architecture: \n\t",normal_arc)
					print("\tTrain amount: \n\t",normal_train_amt, "Total train: ", np.sum(normal_train_amt),"\t Dict size: ", normal_train_dict_length)
					#print("\tLast received dict: \n\t",temp_normal_dict)
					print("\tReduce architecture: \n\t",reduce_arc)
					print("\tTrain amount: \n\t",reduce_train_amt, "Total train: ", np.sum(reduce_train_amt),"\t Dict size: ", reduce_train_dict_length)
					print("\tNormal dict: \n\t",accuracy_scaling.normal_train_dict)
					print("\tReduce dict: \n\t",accuracy_scaling.reduce_train_dict)
					#print("Testing acc scaling with current arc -> ", test_acc_scaling)
					
					

				if actual_step % ops["eval_every"] == 0:
					if (FLAGS.controller_training and
							epoch % FLAGS.controller_train_every == 0):
						print("Epoch {}: Training controller".format(epoch))
						for ct_step in range(FLAGS.controller_train_steps *
											 FLAGS.controller_num_aggregate):
							temp_normal_array, temp_reduce_array = accuracy_scaling.get_dicts_as_numpy_arrays()
							
							
							run_ops = [
								controller_ops["loss"],
								controller_ops["entropy"],
								controller_ops["lr"],
								controller_ops["grad_norm"],
								controller_ops["valid_acc"],
								controller_ops["normal_arc"],
								controller_ops["reduce_arc"],
								controller_ops["scaled_accuracy"],
								controller_ops["baseline"],
								controller_ops["skip_rate"],
								controller_ops["train_op"],
								controller_ops["normal_arc_training"],
								controller_ops["reduce_arc_training"]
							]
							#print("running controller step")
							mov_avg_accuracy = mov_avg_accuracy_struct.get_mov_average()
							mov_avg_training = mov_avg_training_struct.get_mov_average()
							loss, entropy, lr, gn, val_acc, normal_arc, reduce_arc, scaled_acc, bl, skip, _, normal_arc_training, reduce_arc_training = sess.run(
									run_ops,
									feed_dict=
										{"normal_array:0":temp_normal_array, 
										"reduce_array:0":temp_reduce_array,
										"mov_avg_accuracy:0":mov_avg_accuracy,
										"mov_avg_training:0":mov_avg_training})
							controller_step = sess.run(controller_ops["train_step"])
							curr_time = time.time()
							### controller log
							
							mov_avg_accuracy_struct.push(val_acc)
							mov_avg_training_struct.push(np.sum(normal_arc_training)+np.sum(reduce_arc_training))
							
							normal_arc_str = ','.join(['%d' % num for num in normal_arc])
							reduce_arc_str = ','.join(['%d' % num for num in reduce_arc])
							normal_arc_training_str = ','.join(['%d' % num for num in normal_arc_training])
							reduce_arc_training_str = ','.join(['%d' % num for num in reduce_arc_training])
							
							logline = str(epoch)+";"
							logline +=str(controller_step)+";"
							logline +=normal_arc_str+";"
							logline +=reduce_arc_str+";"
							logline +=normal_arc_training_str+";"
							logline +=reduce_arc_training_str+";"
							logline +=str(val_acc)+";"
							logline +="{:10.4f}".format(float(curr_time - start_time))
				
							controller_logfile.write(logline+"\n")
							###
							if ct_step % FLAGS.log_every == 0:
								
								log_string = ""
								log_string += "ctrl_step = {:<6d}".format(controller_step)
								log_string += " loss = {:<7.3f}".format(loss)
								log_string += " ent = {:<5.2f}".format(entropy)
								log_string += "   lr = {:<6.4f}".format(lr)
								log_string += "   |g| = {:<8.4f}".format(gn)
								log_string += " acc = {:<6.4f}".format(val_acc)
								log_string += " mov_avg_acc = {:<6.4f}".format(mov_avg_accuracy)
								#log_string += " s_acc = "
								#log_string += str(scaled_acc)
								log_string += "   bl = {:<5.2f}".format(bl)
								log_string += "  mins = {:<.2f}".format(
									float(curr_time - start_time) / 60)
								print("Controller step #",controller_step,":")
								normal_train_amt = accuracy_scaling.get_trained_arc(normal_arc, "normal")
								reduce_train_amt = accuracy_scaling.get_trained_arc(reduce_arc, "reduce")
								normal_train_dict_length = len(accuracy_scaling.normal_train_dict)
								reduce_train_dict_length = len(accuracy_scaling.reduce_train_dict)
								
								print("\tNormal architecture: \n\t",normal_arc)
								print("\tTrain amount: \n\t",normal_train_amt, "Total train: ", np.sum(normal_train_amt),"\t Dict size: ", normal_train_dict_length)
								print("\tReduce architecture: \n\t",reduce_arc)
								print("\tTrain amount: \n\t",reduce_train_amt, "Total train: ", np.sum(reduce_train_amt),"\t Dict size: ", reduce_train_dict_length)
								print("\tScaled acc: \n\t",scaled_acc)
								print("\tReceived Arc trainings: \n\t", normal_arc_training,"\n\t", reduce_arc_training)
								print(log_string)

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
	child_logfile.close()
	controller_logfile.close()
	
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
