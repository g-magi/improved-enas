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
		

class AccuracyScaler:
	
	def _tf_convert_arc_to_seq(self, arc):
		# tupla con i, output, arc
		loop_tuple = (tf.constant(1), tf.constant(0), arc)
		def _cond(i, output, arc):
			return tf.less(i, tf.shape(arc)[0])
		def _body(i, output, arc):
			current_node = tf.mod(i,4)
			current_item = tf.math.div(i,4)
			if tf.math.equal(current_item,1):
				#x_op
				output = tf.constant(0)
			elif tf.math.equal(current_item,3):
				#y_op
				output = tf.constant(1)
			#((i+1)*10)
			node_const = tf.math.multiply(tf.math.add(i, 1),10)
			#(y_op+1)
			op_const = tf.math.add(tf.gather(arc,i),1)
			const_plus_op = tf.math.add(node_const,op_const)
			const_plus_op = tf.math.multiply(const_plus_op,10)
			output = tf.math.add(const_plus_op,output)
			return tf.math.add(i,2), output, arc
		
		return tf.while_loop(_cond, _body, loop_tuple)[1]
	
	def _tf_get_arc_training(self, arc_seq, current_dict):
		loop_tuple = (tf.constant(0), tf.constant(0), arc_seq, current_dict)
		def _cond(i, output, arc_seq, current_dict):
			return tf.less(i, tf.shape(arc_seq)[0])
		def _body(i, output, arc_seq, current_dict):
			current_op = tf.gather(arc_seq, [i])
			output = current_dict.lookup(current_op)
			return tf.math.add(i,1), output, arc_seq, current_dict
		
		return tf.while_loop(_cond, _body, loop_tuple)[1]
			
	
	def _tf_get_hash_table_from_dict(self, tf_dict):
		loop_tuple = (tf.constant(0), tf.constant(0), tf.constant(0), tf_dict)
		def _cond(i, output_key, output_value, tf_dict):
			return tf.less(i, tf.shape(tf_dict)[0])
		def _body(i, output, tf_dict):
			output_key = tf.gather_nd(tf_dict,[i,0])
			output_value = tf.gather_nd(tf_dict,[i,1])
			i = tf.math.add(i,1)
			return i, output_key, output_value, tf_dict
		loop_outputs = tf.while_loop(_cond,_body, loop_tuple)
		keys = loop_outputs[1]
		values = loop_outputs[2]
		table = tf.lookup.StaticHashTable(
			tf.lookup.KeyValueTensorInitializer(keys, values), -1)
		return table
		
	
	def tf_compute_average_arc_training(self, current_dict):
		tf_full_dict = current_dict.export()
		tf_average = tf.gather(tf.math.reduce_mean(tf_full_dict,0),[1])
		return tf_average
		
		##### TODO TODO TODO
		
	@tf.function
	def tf_get_scaled_accuracy(self, normal_dict, reduce_dict, accuracy, normal_arc, reduce_arc, scaling_method="linear", arc_handling="sum"):
		# reshaping dictionaries in [x, 2] tensors and then putting them in hashtables
		tf_normal_dict = tf.convert_to_tensor(normal_dict)
		tf_normal_dict = tf.reshape(tf_normal_dict, [-1,2])
		tf_normal_dict = _tf_get_hash_table_from_dict(tf_normal_dict)
		tf_reduce_dict = tf.convert_to_tensor(reduce_dict)
		tf_reduce_dict = tf.reshape(tf_reduce_dict, [-1,2])
		tf_reduce_dict = _tf_get_hash_table_from_dict(tf_reduce_dict)
		
		# transforming architectures in sequences of dict keys
		tf_normal_arc_seq = _tf_convert_arc_to_seq(normal_arc)
		tf_reduce_arc_seq = _tf_convert_arc_to_seq(reduce_arc)
		
		
		# transforming sequences of dict keys into sequences of training amounts
		tf_normal_arc_training = _tf_get_arc_training(tf_normal_arc_seq)
		tf_reduce_arc_training = _tf_get_arc_training(tf_reduce_arc_seq)
		
		# sum of the training amounts
		tf_normal_arc_training_sum = tf.reduce_sum(tf_normal_arc_training)
		tf_reduce_arc_training_sum = tf.reduce_sum(tf_reduce_arc_training)
		
		combined_arcs_training = tf.constant(0.0)
		if arc_handling is "sum":
			combined_arcs_training = tf_normal_arc_training_sum + tf_reduce_arc_training_sum
		elif arc_handling is "avg":
			combined_arcs_training = (tf_normal_arc_training_sum + tf_reduce_arc_training_sum)//2
		tf.cast(combined_arcs_training, tf.float64)
		
		scaled_accuracy = tf.constant(5.0)
		
		if scaling_method is "linear":
			scaled_accuracy = accuracy * combined_arcs_training
		elif scaling_method is "compare_avg":
			average_normal_arc_training = tf_compute_average_arc_training(normal_dict)
			average_reduce_arc_training = tf_compute_average_arc_training(reduce_dict)
			average_arc_training = tf.constant(5.0)
			if arc_handling is "sum":
				average_arc_training = average_normal_arc_training+average_reduce_arc_training
			elif arc_handling is "avg":
				average_arc_training = (average_normal_arc_training+average_reduce_arc_training)//2
			tf.cast(average_arc_training, tf.float64)
			scaling_factor = average_arc_training/combined_arcs_training
			scaled_accuracy = accuracy*scaling_factor
		
		return scaled_accuracy
	"""
	def get_scaled_accuracy(self, normal_dict, reduce_dict ,accuracy, normal_arc, reduce_arc, scaling_method="linear", arc_handling="sum"):
		if tf.rank(normal_dict) is 0:
			return 5, normal_dict, reduce_dict
		#if type(normal_arc) is not np.ndarray:
		#	return 0.0
		normal_dict, reduce_dict = self.convert_numpy_arrays_to_dicts(normal_dict,reduce_dict)
		normal_arc_training = self.get_trained_arc(normal_arc, "normal")
		reduce_arc_training = self.get_trained_arc(reduce_arc, "reduce")
		normal_arc_training = np.sum(normal_arc_training)
		reduce_arc_training = np.sum(reduce_arc_training)
		combined_arcs_training = 0.0
		
		## arc handling section
		if arc_handling is "sum":
			combined_arcs_training = normal_arc_training + reduce_arc_training
		elif arc_handling is "avg":
			combined_arcs_training = (normal_arc_training+reduce_arc_training)//2 #i use // so it gets floored
		
		scaled_accuracy = 5.0
		
		## scaling section
		if scaling_method is "linear":
			scaled_accuracy = accuracy * float(combined_arcs_training)
		elif scaling_method is "compare_avg":
			average_normal_arc_training = self._compute_average_arc(len(normal_arc)//4, "normal")
			average_normal_arc_training = np.avg(average_normal_arc_training)
			average_reduce_arc_training = self._compute_average_arc(len(normal_arc)//4, "reduce")
			average_reduce_arc_training = np.avg(average_reduce_arc_training)
			average_arc_training = 0
			if arc_handling is "sum":
				average_arc_training = (average_normal_arc_training+average_reduce_arc_training)
			elif arc_handling is "avg":
				average_arc_training = (average_normal_arc_training+average_reduce_arc_training)//2
			
			scaling_factor = float(average_arc_training)/float(combined_arcs_training)
			scaled_accuracy = accuracy * scaling_factor
		
		##
		
		#return scaled_accuracy, self._get_dict_as_numpy_array("normal"), self._get_dict_as_numpy_array("reduce")
		return scaled_accuracy, normal_dict, reduce_dict
	
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
	
	def convert_numpy_array_to_dict(self,array):
		temp_dict = {}
		if array.shape[0] is None:
			return temp_dict
		for i in range(array.shape[0]//4):
			x_key = array[i*4+0]
			x_value = array[i*4+1]
			y_key = array[i*4+2]
			y_value = array[i*4+3]
			temp_dict[x_key] = x_value
			temp_dict[y_key] = y_value
		return temp_dict
	
	def _set_numpy_array_as_dict(self,dict_type, array):
		temp_dict = self.convert_numpy_array_to_dict(array)
		
		
		if dict_type is "normal":
			self.normal_train_dict.clear()
			for key, value in temp_dict.items():
				if not tf.is_tensor(value):
					self.normal_train_dict[key] = value
			return self._get_dict_as_numpy_array("normal")
		elif dict_type is "reduce":
			self.reduce_train_dict.clear()
			for key, value in temp_dict.items():
				if not tf.is_tensor(value):
					self.reduce_train_dict[key] = value
			return self._get_dict_as_numpy_array("reduce")
		
	
	def convert_numpy_arrays_to_dicts(self, normal_array, reduce_array):
		normal_array = self._set_numpy_array_as_dict(dict_type="normal", array=normal_array)
		reduce_array = self._set_numpy_array_as_dict(dict_type="reduce", array=reduce_array)
		return normal_array, reduce_array
	"""
	#
	