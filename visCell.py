import pygraphviz as pgv
import numpy as np
import sys
import csv
import json
import ast

def construct_block(graph, num_block, ops):

	ops_name = ["conv 3x3", "conv 5x5", "avg pool", "max pool", "identity", "add", "concat"]

	for i in range(0, 2):
		graph.add_node(num_block*10+i+1,
					   label="{}".format(ops_name[ops[2*i+1]]),
					   color='black',
					   fillcolor='yellow',
					   shape='box',
					   style='filled')

	#graph.add_subgraph([num_block*10+1, num_block*10+2], rank='same')

	graph.add_node(num_block*10+3,
				   label="Add",
				   color='black',
				   fillcolor='greenyellow',
				   shape='box',
				   style='filled')

	graph.add_subgraph([num_block*10+1, num_block*10+2, num_block*10+3],
					   name='cluster_s{}'.format(num_block))

	for i in range(0, 2):
		graph.add_edge(num_block*10+i+1, num_block*10+3)

def connect_block(graph, num_block, ops, output_used):

	for i in range(0, 2):
		graph.add_edge(ops[2*i]*10+3, (num_block*10)+i+1)
		output_used.append(ops[2*i]*10+3)

def creat_graph(cell_arc):

	G = pgv.AGraph(directed=True, strict=True, fontname='Helvetica', arrowtype='open', rankdir='TD')

	#creat input
	G.add_node(3, label="H[i-1]", color='black', shape='box')
	G.add_node(13, label="H[i]", color='black', shape='box')
	G.add_subgraph([3, 13], name='cluster_inputs', rank='same', rankdir='TD', color='white')

	#creat blocks
	for i in range(0, len(cell_arc)):
		construct_block(G, i+2, cell_arc[i])

	#connect blocks to each other
	output_used = []
	for i in range(0, len(cell_arc)):
		connect_block(G, i+2, cell_arc[i], output_used)

	#creat output
	G.add_node((len(cell_arc)+2)*10+3,
			   label="Concat",
			   color='black',
			   fillcolor='pink',
			   shape='box',
			   style='filled')

	for i in range(0, len(cell_arc)+2):
		if not(i*10+3 in output_used) :
			G.add_edge(i*10+3, (len(cell_arc)+2)*10+3)
	
	
	G.add_node((len(cell_arc)+2)*10+4,
			   label="conv 1x1",
			   color='black',
			   fillcolor='yellow',
			   shape='box',
			   style='filled')
	G.add_edge((len(cell_arc)+2)*10+3, (len(cell_arc)+2)*10+4)
	
	G.add_node((len(cell_arc)+2)*10+5,
			   label="H[i+1]",
			   color='black',
			   fillcolor='white',
			   shape='box',
			   style='filled')
			   
	G.add_edge((len(cell_arc)+2)*10+4, (len(cell_arc)+2)*10+5)
	return G

#### CREATING GRAPH FOR ARCHITECTURE

def create_arc_graph(arc_dict):
	dict_len = len(arc_dict)
	G = pgv.AGraph(directed=True, strict=True, fontname='Helvetica', arrowtype='open', rankdir='TD')
	
	#input node
	G.add_node(0,
		label="Input",
		color = 'black',
		fillcolor = 'white',
		shape = 'box',
		style = 'filled')
	
	for i in range(1, dict_len+1):
		current_layer_name = "layer_"+str(i-1)
		current_dims = arc_dict[current_layer_name]["dims"]
		current_type = arc_dict[current_layer_name]["type"]
		current_layer_label = current_layer_name
		current_layer_dims = str(current_dims[0])+"x"+str(current_dims[1])+"x"+str(current_dims[2])
		current_layer_label += "\n"
		current_layer_label += current_layer_dims
		current_layer_label += "\n"
		current_layer_label += current_type+" layer"
		
		fillcolor = 'coral'
		shape = 'box'
		if current_type == "reduce":
			fillcolor = 'lightskyblue'
			shape = 'box'
		G.add_node(i,
			label=current_layer_label,
			color = 'black',
			fillcolor = fillcolor,
			shape = shape,
			style = 'filled')
		G.add_edge(i-1, i)
	
	G.add_node(dict_len+1,
		label = 'Output',
		color = 'black',
		fillcolor = 'white',
		shape = 'box',
		style = 'filled')
	G.add_edge(dict_len, dict_len+1)
	return G

#!python visCell.py "1 3 0 0 2 1 0 0 1 1 1 0 1 4 0 0 2 0 1 4 0 0 1 2 0 1 0 4 1 0 1 1 1 4 0 1 0 1 0 0"
# 1,2,1,4,0,1,1,4,0,1,1,3,0,3,1,1,1,1,0,0
# 1,2,1,1,0,3,0,3,0,4,1,4,0,1,3,1,1,1,0,2
def main():

	if(len(sys.argv) < 3):
		norm_cell = "1 2 1 4 0 1 1 4 0 1 1 3 0 3 1 1 1 1 0 0"
		redu_cell = "1 2 1 1 0 3 0 3 0 4 1 4 0 1 3 1 1 1 0 2"
		arc_info_file = None
		arc_dict = None
	else:
		"""
		norm_cell, redu_cell = "", ""
		for i in range(1, len(sys.argv)/2+1):
			norm_cell += "{} ".format(sys.argv[i])
		for i in range(len(sys.argv)/2+1, len(sys.argv)):
			redu_cell += "{} ".format(sys.argv[i])
		print("{}\n{}".format(norm_cell, redu_cell))
		"""
		arcs_file = sys.argv[1]
		arc_info_file = sys.argv[2]
		with open(arcs_file) as f:
			reader = csv.reader(f, delimiter=";")
			best_arcs = next(reader)
			norm_cell = ast.literal_eval(best_arcs[0])
			redu_cell = ast.literal_eval(best_arcs[1])
			print("norm_cell: ", norm_cell)
			print("redu_cell: ", redu_cell)
			print("arc accuracy: ", best_arcs[2])
		
		with open(arc_info_file) as f:
			arc_dict = json.load(f)
			#print(json.dumps(arc_dict, indent=4, sort_keys=True))
			

	#ncell = np.array([int(x) for x in norm_cell.split(" ") if x])
	#rcell = np.array([int(x) for x in redu_cell.split(" ") if x])
	ncell = np.array(norm_cell)
	rcell = np.array(redu_cell)

	ncell = np.reshape(ncell, [-1, 4])
	rcell = np.reshape(rcell, [-1, 4])

	Gn = creat_graph(ncell)
	Gr = creat_graph(rcell)
	Ga = None
	if arc_dict is not None:
		Ga = create_arc_graph(arc_dict)
		Ga.write("arc.dot")
		vizGa = pgv.AGraph("arc.dot")
		vizGa.layout(prog='dot')
		vizGa.draw("arc.png")

	Gn.write("ncell.dot")
	Gr.write("rcell.dot")

	vizGn = pgv.AGraph("ncell.dot")
	vizGr = pgv.AGraph("rcell.dot")

	vizGn.layout(prog='dot')
	vizGr.layout(prog='dot')

	vizGn.draw("ncell.png")
	vizGr.draw("rcell.png")


if __name__ == '__main__':
	main()
