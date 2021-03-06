import numpy as np
import tensorflow.compat.v1 as tf

class ArchitectureTrainingStorage:
	normal_train_dict = None
	reduce_train_dict = None
	def __init__(self):
		if self.normal_train_dict is None:
			self.normal_train_dict = {}
		if self.reduce_train_dict is None:
			self.reduce_train_dict = {}
		
	def _split_arc_seq(self,arc_seq):
		arc_seq_length = 0
		if type(arc_seq) is np.ndarray:
			arc_seq_length = arc_seq.shape[0]
		else:
			arc_seq_length = arc_seq.get_shape().as_list()[0]
		assert arc_seq_length != 0
		arc_nodes_amt = arc_seq_length//4
		arc_nodes = [np.ones(4)] * arc_nodes_amt
		if type(arc_seq) is np.ndarray:
			arc_nodes = np.split(arc_seq, arc_nodes_amt)
		return arc_nodes
	
	def _save_trained_op(self,key, arc_type):
		if arc_type is "normal":
			if key in self.normal_train_dict:
				self.normal_train_dict[key]+=1
			else:
				self.normal_train_dict[key]=1
		elif arc_type is "reduce":
			if key in self.reduce_train_dict:
				self.reduce_train_dict[key]+=1
			else:
				self.reduce_train_dict[key]=1
	
	def save_trained_arc(self,arc_seq, arc_type):
		arc_nodes = self._split_arc_seq(arc_seq)
		for i, node in enumerate(arc_nodes):
			x_op = node[1]
			y_op = node[3]
			#x_key = "node"+str(i)+"_x_op"+str(x_op)
			x_key = (((i+1)*10)+(x_op+1))*(10)+0
			#y_key = "node"+str(i)+"_y_op"+str(y_op)
			y_key = (((i+1)*10)+(y_op+1))*(10)+1
			self._save_trained_op(x_key, arc_type)
			self._save_trained_op(y_key, arc_type)
	
	def _get_trained_op(self,key, arc_type):
		if arc_type is "normal":
			if key in self.normal_train_dict:
				return self.normal_train_dict[key]
			else:
				return -10
		elif arc_type is "reduce":
			if key in self.reduce_train_dict:
				return self.reduce_train_dict[key]
			else:
				return -10
	
	# returns a numpy array of [nodes_amt*5] items that has the same structure as the return from [get_trained_arc]
	
	def _compute_average_arc(self,nodes_amt, arc_type):
		average_arc = np.zeros(nodes_amt*2, dtype=int)
		for i in range(nodes_amt):
			#per ogni nodo
			#guardo l'op x
			x_total = 0
			#per ogni op x di ogni nodo sommo il training di tutte le operazioni
			for i2 in range(5):
				key = "node"+str(i)+"_x_op"+str(i2)
				if arc_type is "normal" and key in self.normal_train_dict:
					x_total += self.normal_train_dict[key]
				elif arc_type is "reduce" and key in self.reduce_train_dict:
					x_total += self.reduce_train_dict[key]
			y_total = 0
			#per ogni op y di ogni nodo sommo il training di tutte le operazioni
			for i2 in range(5):
				key = "node"+str(i)+"_y_op"+str(i2)
				if arc_type is "normal" and key in self.normal_train_dict:
					y_total += self.normal_train_dict[key]
				elif arc_type is "reduce" and key in self.reduce_train_dict:
					y_total += self.reduce_train_dict[key]
				
			average_arc[i*2]=x_total//5
			average_arc[(i*2)+1]=y_total//5
		
		return average_arc
	
	def get_trained_arc(self,arc_seq, arc_type):
		arc_nodes = self._split_arc_seq(arc_seq)
		trained_arc = []
		for i, node in enumerate(arc_nodes):
			x_op = node[1]
			y_op = node[3]
			#x_key = "node"+str(i)+"_x_op"+str(x_op)
			x_key = (((i+1)*10)+(x_op+1))*(10)+0
			#y_key = "node"+str(i)+"_y_op"+str(y_op)
			y_key = (((i+1)*10)+(y_op+1))*(10)+1
			x_train_amt = self._get_trained_op(x_key, arc_type)
			y_train_amt = self._get_trained_op(y_key, arc_type)
			trained_arc.append(x_train_amt)
			trained_arc.append(y_train_amt)
		
		return trained_arc
	
	def _get_dict_as_numpy_array(self, dict_type):
		
		out_list = []
		if dict_type is "normal":
			for key, value in self.normal_train_dict.items():
				out_list.append(key)
				out_list.append(value)
		if dict_type is "reduce":
			for key, value in self.reduce_train_dict.items():
				out_list.append(key)
				out_list.append(value)
		
		out_array = np.asarray(out_list)
		return out_array
	
	def get_dicts_as_numpy_arrays(self):
		out_normal = self._get_dict_as_numpy_array("normal")
		out_reduce = self._get_dict_as_numpy_array("reduce")
		return out_normal, out_reduce

class AccuracyScaler:

	"""

	def _tf_convert_arc_to_seq(self, arc_):
		arc_short = arc_[1::2]
		seq = tf.TensorArray(tf.int32, size = 0, dynamic_size = True)
		i = tf.constant(0)
		loop_tuple = (i, seq, arc_short)
		def _cond(i, seq, arc):
			return tf.math.less(i,tf.shape(arc)[0])
		def _body(i, seq, arc):
			def _body_x():
				return tf.constant(0)
			def _body_y():
				return tf.constant(1)
			x_or_y = tf.math.equal(tf.math.floormod(i, 2),0)
			op_x_or_y = tf.cond(x_or_y,_body_x,_body_y)
			node_id = tf.math.add(tf.math.floordiv(i, 2),1)
			node_id = tf.math.multiply(node_id,100)
			op_id = tf.gather(arc, i)
			op_id = tf.math.add(op_id,1)
			op_id = tf.math.multiply(op_id,10)
			key = node_id+op_id+op_x_or_y
			# return vars
			seq = seq.write(i,key)
			i = tf.math.add(i, 1)
			return i, seq, arc
		
		seq = tf.while_loop(_cond, _body, loop_tuple)[1]
		seq = tf.reshape(seq.stack(),[-1])
		return seq
		
	def _tf_get_arc_training(self, arc_seq, current_dict):
		output = tf.TensorArray(tf.int32, size = 0, dynamic_size=True)
		loop_tuple = (tf.constant(0), output, arc_seq)
		def _cond(i, output, arc_seq):
			return tf.math.less(i, tf.shape(arc_seq)[0])
		def _body(i, output, arc_seq):
			current_op = tf.gather(arc_seq, [i])
			output = output.write(i,current_dict.lookup(current_op))
			#output = output.write(i,10)
			return tf.math.add(i,1), output, arc_seq
		
		output = tf.while_loop(_cond, _body, loop_tuple)
		output = output[1]
		output = output.stack()
		output = tf.reshape(output,[-1])
		return output
			

	def _tf_get_hash_table_from_dict(self, tf_dict):
		i = tf.constant(0)
		output_key = tf.TensorArray(tf.int32, size = 0, dynamic_size=True)
		output_value = tf.TensorArray(tf.int32, size = 0, dynamic_size=True)
		loop_tuple = (i, output_key, output_value, tf_dict)
		def _cond(i, output_key, output_value, tf_dict):
			return tf.math.less(i, tf.shape(tf_dict)[0])
		def _body(i, output_key, output_value, tf_dict):
			key = tf.gather_nd(tf_dict,[i,0])
			output_key = output_key.write(i,key)
			value = tf.gather_nd(tf_dict,[i,1])
			output_value = output_value.write(i,value)
			i = tf.math.add(i,1)
			return i, output_key, output_value, tf_dict
		loop_outputs = tf.while_loop(_cond,_body, loop_tuple)
		keys = loop_outputs[1].stack()
		keys = tf.reshape(keys, [-1])
		values = loop_outputs[2].stack()
		values = tf.reshape(values, [-1])
		table = tf.lookup.StaticHashTable(tf.lookup.KeyValueTensorInitializer(keys, values), -1)
		return table
		

	def tf_compute_average_arc_training(self, current_dict):
		tf_full_dict = current_dict.export()[1]
		tf_average = tf.reduce_mean(tf_full_dict)
		return tf_average
		
	"""
		
	@tf.function
	def tf_get_scaled_accuracy(self,
				normal_dict,							#dictionary for normal architecture
				reduce_dict,							#dictionary for reduce architecture
				accuracy,								#accuracy for current architecture, computed on a batch of valid data
				normal_arc,								#current normal architecture
				reduce_arc,								#current reduce architecture
				mov_avg_accuracy,						#moving average of the accuracy of some (decided in 'main_controller_child_trainer.py') of the last few architectures
				mov_avg_training,						#moving average of the training amount of some (same as above) of the last few architectures
				scaling_method=tf.constant("linear"),	#scaling method chosen
				arc_handling=tf.constant("sum"),		#how to handle the values for the 2 different architectures (either 'sum' or 'avg')
				):
		# initializing constants
		
		###TODO: make these into placeholders so they can be initialized remotely
		_scaling_factor_threshold_low_medium = tf.constant(0.7, tf.float32)
		_scaling_factor_threshold_medium_high = tf.constant(1.1, tf.float32)
	
	
		# reshaping dictionaries in [x, 2] tensors and then putting them in hashtables

		
		scaled_accuracy = tf.constant(-10.0)
		"""
		## linear scaling
			# multiplies accuracy of arc by its total training
			# "linear"
		def _scale_linear():
			return combined_arcs_training
		## average scaling
			# multiplies accuracy of arc by a scaling factor
			# the scaling factor is [average training]/[arc training]
			# "average"
		def _scale_avg_sum():
			return tf.math.add(average_normal_arc_training,average_reduce_arc_training)
		def _scale_avg_avg():
			return tf.math.floordiv(_scale_avg_sum(), 2)
		def _scale_avg():
			average_arc_training = tf.cond(tf.math.equal(arc_handling,tf.constant("sum")),_scale_avg_sum,_scale_avg_avg)
			average_arc_training = tf.cast(average_arc_training, tf.float32)
			scaling_factor = average_arc_training/combined_arcs_training
			return scaling_factor
		"""
		## no scaling
			# "none"
		def _scale_none():
			return tf.constant(1.0,tf.float32)
		##
		## moving average scaling
			# multiplies accuracy of arc by a scaling factor
			# the scaling factor is [mov_avg_training]/[arc training]
			# "average"
		"""
		def _scale_mov_avg():
			def _case_zero():
				return tf.constant(1.0, tf.float32)
			def _case_default():
				# it's yuuuge
				tf.tables_initializer()
				tf_normal_dict = tf.convert_to_tensor(normal_dict)
				tf_normal_dict = tf.reshape(tf_normal_dict, [-1,2])
				tf_normal_dict = self._tf_get_hash_table_from_dict(tf_normal_dict)
				tf_reduce_dict = tf.convert_to_tensor(reduce_dict)
				tf_reduce_dict = tf.reshape(tf_reduce_dict, [-1,2])
				tf_reduce_dict = self._tf_get_hash_table_from_dict(tf_reduce_dict)
				
				# transforming architectures in sequences of dict keys
				tf_normal_arc_seq = self._tf_convert_arc_to_seq(normal_arc)
				tf_reduce_arc_seq = self._tf_convert_arc_to_seq(reduce_arc)
				
				
				# transforming sequences of dict keys into sequences of training amounts
				tf_normal_arc_training = self._tf_get_arc_training(tf_normal_arc_seq, tf_normal_dict)
				tf_reduce_arc_training = self._tf_get_arc_training(tf_reduce_arc_seq, tf_reduce_dict)
				
				# sum of the training amounts
				tf_normal_arc_training_sum = tf.reduce_sum(tf_normal_arc_training)
				tf_reduce_arc_training_sum = tf.reduce_sum(tf_reduce_arc_training)
				
				combined_arcs_training = tf.constant(-10.0)
				def _cat_sum(): 
					return tf.math.add(tf_normal_arc_training_sum,tf_reduce_arc_training_sum)
				def _cat_avg(): 
					temp_combined_arcs_training = tf.math.add(tf_normal_arc_training_sum,tf_reduce_arc_training_sum)
					temp_combined_arcs_training = tf.math.floordiv(temp_combined_arcs_training, 2)
					return tf.cast(temp_combined_arcs_training, tf.int32)
				
				combined_arcs_training = tf.cond(tf.math.equal(arc_handling,tf.constant("sum")),_cat_sum,_cat_avg)
				
				average_normal_arc_training = self.tf_compute_average_arc_training(tf_normal_dict)
				average_reduce_arc_training = self.tf_compute_average_arc_training(tf_reduce_dict)
				combined_arcs_training = tf.cast(combined_arcs_training, tf.float32)
				
				# end of yuuuge stuff
				return mov_avg_training/combined_arcs_training
			scaling_factor = tf.cond(tf.math.equal(mov_avg_training,tf.constant(0.0)),
								_case_zero,
								_case_default)
			return scaling_factor
		"""
		## greedy average scaling
			# multiplies accuracy of arc by a scaling factor
			# the scaling factor is [arc training]/[average training]
			# "greedy-average"
		"""
		def _scale_greedy_avg_sum():
			return tf.math.add(average_normal_arc_training,average_reduce_arc_training)
		def _scale_greedy_avg_avg():
			return tf.math.floordiv(_scale_avg_sum(), 2)
		
		def _scale_greedy_avg():
			average_arc_training = tf.cond(
				tf.math.equal(arc_handling,tf.constant("sum")),
				_scale_greedy_avg_sum,
				_scale_greedy_avg_avg)
			
			average_arc_training = tf.cast(average_arc_training, tf.float32)
			scaling_factor = combined_arcs_training/average_arc_training
			return scaling_factor
		##
		"""
		## greedy accuracy scaling
			# multiplies accuracy of arc by a scaling factor
			# the scaling factor depends on [current accuracy] compared
			# to [moving average accuracy], boosting better architectures
		
		def _scale_greedy_accuracy():
			def _case_zero():
				return tf.constant(1.0, tf.float32)
			def _case_default():
				return accuracy/mov_avg_accuracy
			scaling_factor = tf.cond(tf.math.equal(mov_avg_accuracy,tf.constant(0.0)),
								_case_zero,
								_case_default)
			return scaling_factor
		
		##
		
		## combined accuracy scaling
			# multiplies accuracy of arc by a scaling factor
			# the scaling factor depends on [current accuracy] compared
			# to [moving average accuracy], boosting better architectures
			# also 
		
		def _scale_combined():
			
			scaling_factor_acc = _scale_greedy_accuracy()
			
			#scaling_factor_train = _scale_mov_avg()
			#scaling_factor_train = tf.constant(1.0, tf.float32)
			
			def _case_factor_high():
				return scaling_factor_acc
			
			#def _case_factor_medium():
			#	return (scaling_factor_train+scaling_factor_acc)/2
			def _case_factor_medium():
				return _case_factor_high()
			
			
			def _case_factor_low():
				return tf.constant(0.01, tf.float32)
		
			#_scaling_factor_threshold_low_medium
			#_scaling_factor_threshold_medium_high
			is_factor_high = tf.math.greater(scaling_factor_acc,_scaling_factor_threshold_medium_high)
			
			is_factor_low = tf.math.less(scaling_factor_acc,_scaling_factor_threshold_low_medium)
			
			is_factor_medium = tf.math.logical_and(
								tf.math.logical_not(is_factor_high),
								tf.math.logical_not(is_factor_low)
															)
			
			scaling_factor = tf.case([
					(is_factor_high,_case_factor_high),
					(is_factor_low,_case_factor_low),
					(is_factor_medium,_case_factor_medium)
				],default = _case_factor_medium)
				
			return scaling_factor
		
		##
		
		###
		#linear_case 		= tf.constant("linear")
		#average_case 		= tf.constant("average")
		#moving_average_case = tf.constant("moving-average")
		none_case 			= tf.constant("none")
		#greedy_average_case = tf.constant("greedy-average")
		greedy_accuracy_case = tf.constant("greedy-accuracy")
		combined_case 		= tf.constant("combined")
		scaling_factor = tf.case([
				#(tf.math.equal(scaling_method,linear_case),_scale_linear),
				#(tf.math.equal(scaling_method,average_case),_scale_avg),
				#(tf.math.equal(scaling_method,moving_average_case),_scale_mov_avg),
				(tf.math.equal(scaling_method,none_case),_scale_none),
				#(tf.math.equal(scaling_method,greedy_average_case),_scale_greedy_avg),
				(tf.math.equal(scaling_method,greedy_accuracy_case),_scale_greedy_accuracy),
				(tf.math.equal(scaling_method,combined_case),_scale_combined)
				],default = _scale_none, exclusive = True)
		
		scaled_accuracy = accuracy*scaling_factor
		
		return scaled_accuracy, accuracy, scaling_factor