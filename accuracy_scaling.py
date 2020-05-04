class AccuracyScaling():
	def __init__(self, normal_train_dict={}, reduce_train_dict={}):
		self.normal_train_dict = normal_train_dict
		self.reduce_train_dict = reduce_train_dict
		
	
	def _split_arc_seq(arc_seq):
		arc_seq_length = len(arc_seq)
		arc_nodes_amt = FLAGS.child_num_cells
		assert arc_seq_length%arc_nodes_amt == 0
		arc_nodes = np.split(arc_seq, arc_nodes_amt)
		return arc_nodes
	
	
	def _save_trained_op(key, arc_type):
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
	
	def save_trained_arc(arc_seq, arc_type):
		arc_nodes = _split_arc_seq(arc_seq)
		for i, node in enumerate(arc_nodes):
			x_op = node[1]
			y_op = node[3]
			x_key = "node"+str(i)+"_x_op"+str(x_op)
			y_key = "node"+str(i)+"_y_op"+str(y_op)
			_save_trained_op(x_key, arc_type)
			_save_trained_op(y_key, arc_type)
		
	def _get_trained_op(key, arc_type):
		if arc_type is "normal":
			if key in self.normal_train_dict:
				return self.normal_train_dict[key]
			else:
				return 0
		elif arc_type is "reduce":
			if key in self.reduce_train_dict:
				return self.reduce_train_dict[key]
			else:
				return 0
			
	def get_trained_arc(arc_seq, arc_type):
		arc_nodes = _split_arc_seq(arc_seq)
		trained_arc = []
		for i, node in enumerate(arc_nodes):
			x_op = node[1]
			y_op = node[3]
			x_key = "node"+str(i)+"_x_op"+str(x_op)
			y_key = "node"+str(i)+"_y_op"+str(y_op)
			x_train_amt = _get_trained_op(x_key, arc_type)
			y_train_amt = _get_trained_op(y_key, arc_type)
			trained_arc.append(x_train_amt)
			trained_arc.append(y_train_amt)
		
		return trained_arc
		
	