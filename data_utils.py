import os
import sys
import random
import cv2
import csv
import random
import numpy as np
import itertools
from sklearn import model_selection as ms
from glob import *
import utils
from utils import plot_data_label
import tensorflow.compat.v1 as tf

random.seed(random.randint(0, 2 ** 31 - 1))

def _read_data(data_path, channel, img_size, n_aug_img):

	global ccc
	global mean
	global std

	ccc = channel

	if n_aug_img == 1:
		aug_flag = False
	else:
		aug_flag = True

	class_list = os.listdir(data_path)
	class_list.sort()
	n_classes = len(class_list)
	images= []
	image_holder = []
	labels= []
	label_holder = []

	for i in range(n_classes):
		img_class = glob(os.path.join(data_path,class_list[i]) + '/*.png')
		images += img_class
		for j in range(len(img_class)):
			labels += [i]

	if channel == "1":
		flags = 0
	else:
		flags = 1

	length_data = len(images)
	bar_size=20
	print("\n")
	for j in range(length_data):
		img = cv2.imread(images[j],flags = flags)
		if j%500==0:
			print("\rreading image number: "+str(j)+"/"+str(length_data)+" <", end="")
			for i_bar in range(bar_size):
				if (j/length_data)*100>=(100/bar_size)*i_bar: 
					print("*",end="")
				else:
					print("-",end="")
			print(">",end="")
		if img is None:
			print(j,"is None, path is: ",images[j])
		if channel ==1:
			img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
			img = cv2.resize(img,(img_size,img_size))

			# Augmentation
			if aug_flag is True:
				for k in range(n_aug_img - 1):
					image_aug = img_augmentation(img)
					image_aug = np.reshape(image_aug, [1,img_size, img_size, channel])
					image_holder.append(image_aug)
					label_holder.append(labels[j])

			img = np.reshape(img, [1, img_size, img_size, channel])

			image_holder.append(img)
			label_holder.append(labels[j])

		else:
			img = cv2.resize(img,(img_size,img_size))

			if aug_flag is True:
				for k in range(n_aug_img - 1):
					image_aug = img_augmentation(img)
					image_aug = np.reshape(image_aug, [1,img_size, img_size, channel])
					image_holder.append(image_aug)
					label_holder.append(labels[j])

			img = np.reshape(img, [1,img_size,img_size,channel])

			image_holder.append(img)
			label_holder.append(labels[j])
	print("")
	
	image_holder =np.concatenate(image_holder, axis = 0)
	label_holder = np.asarray(label_holder, dtype = np.int32)

	if aug_flag is True:
		images = []
		labels = []

		for w in range(n_aug_img):
			holder = []
			total_data = length_data*n_aug_img
			quota = (length_data//n_classes)
			interval = total_data//n_classes
			for r in range(n_classes):
				temp = np.add(np.full((quota), interval*r + quota*w),np.random.permutation(quota))
				holder.extend(temp)

			_ = random.shuffle(holder)
			images.append(image_holder[holder])
			labels.append(label_holder[holder])

		images = np.concatenate(images, axis = 0)
		labels = np.concatenate(labels, axis = 0)

	else:
		idx = np.random.permutation(length_data)
		images = image_holder[idx]
		labels = label_holder[idx]

	n_batch_mean = len(images)
	mean = 0
	std = 0
	for b in range(n_batch_mean):
		mean += np.mean(images[b], axis = (0,1,2))/n_batch_mean
		std += np.std(images[b], axis = (0,1,2))/n_batch_mean

	plot_data_label(images[0:64], labels[0:64],channel ,8,8,8)
	print(data_path)
	print("Mean:", mean)
	print("Std:", std)
	print("_____________________________")

	images = ((images - mean)/std).astype(np.float32)
	return images, labels


def read_data(train_dir,val_dir, test_dir, channel, img_size, n_aug_img):
	'''
	channel = channels of images. MNIST: channels = 1
								Cifar 10: channels = 3
	'''
	print("-"*80)
	print("Reading data")

	images, labels = {}, {}

	images["train"], labels["train"] = _read_data(train_dir, channel, img_size,n_aug_img)
	images["valid"], labels["valid"] = _read_data(val_dir, channel, img_size, 1)
	images["test"], labels["test"] = _read_data(test_dir, channel, img_size, 1)

	return images, labels

def img_augmentation(image):
	global ccc

	def gaussian_noise(image):
		image = image.astype(np.float32)
		size = np.shape(image)
		for i in range(size[0]):
			for j in range(size[1]):
				if ccc == 1:
					q = random.random()
					if q < 0.1:
						image[i][j] = 0
				else:
					q = random.random()
					if q < 0.1:
						image[i][j][:] = 0

		return image.astype(np.uint8)

	def Flip(image):
		img_filped = cv2.flip(image, 1)

		return img_filped

	def enlarge(image, magnification):

		H_before = np.shape(image)[1]
		center = H_before // 2
		M = cv2.getRotationMatrix2D((center, center), 0, magnification)
		img_croped = cv2.warpAffine(image, M, (H_before, H_before))

		return img_croped

	def rotation(image):
		H_before = np.shape(image)[1]
		center = H_before // 2
		angle = random.randint(-20, 20)
		M = cv2.getRotationMatrix2D((center, center), angle, 1)
		img_rotated = cv2.warpAffine(image, M, (H_before, H_before))

		return img_rotated

	def random_bright_contrast(image):
		alpha = random.uniform(1, 0.1)  # for contrast
		alpha = np.minimum(alpha, 1.3)
		alpha = np.maximum(alpha, 0.7)
		beta = random.uniform(32, 6)  # for brightness

		# g(i,j) = alpha*f(i,j) + beta
		img_b_c = cv2.multiply(image, np.array([alpha]))
		img_b_c = cv2.add(img_b_c, beta)

		return img_b_c

	def aug(image, idx):
		augmentation_dic = {0: enlarge(image, 1.2),
							1: rotation(image),
							2: random_bright_contrast(image),
							3: gaussian_noise(image),
							4: Flip(image)}

		image = augmentation_dic[idx]
		return image

	if ccc == 3:  # c is number of channel
		l = 5
	else:
		l = 4

	p = [random.random() for m in range(l)]  # 4 is number of augmentation operation
	for n in range(l):
		if p[n] > 0.50:
			image = aug(image, n)

	return image

### Metodi per il caricamento di dataset keras

def _prepare_dataset_data(num_classes,X_train,y_train,X_VT,y_VT):

	# from 2d to 1d
	y_train = y_train.reshape(-1)
	y_VT = y_VT.reshape(-1)
	
	if X_train.ndim is 3:
		X_train = np.expand_dims(X_train, axis=3)
	if X_VT.ndim is 3:
		X_VT = np.expand_dims(X_VT, axis=3)
	
	# Check whether the image size is a multiple of 2 (unfortunately needed thanks to how enas works)
	image_size = X_train.shape[1]
	if image_size == 28:
		X_train = np.pad(X_train, ((0,0),(2,2),(2,2),(0,0)), mode='edge')
		X_VT = np.pad(X_VT, ((0,0),(2,2),(2,2),(0,0)), mode='edge')
		
	
	# from 0-255 to 0-1
	
	X_train = np.divide(X_train, 255)
	X_VT = np.divide(X_VT, 255)
	
	# split off valid and test
	X_test, X_validation, y_test, y_validation= ms.train_test_split(X_VT, y_VT, test_size=0.3, random_state=1)
	print("X_train shape: ", X_train.shape)
	print("X_test shape: ", X_test.shape)
	print("X_validation shape: ", X_validation.shape)
	print("y_train shape: ", y_train.shape)
	print("y_test shape: ", y_test.shape)
	print("y_validation shape: ", y_validation.shape)
	
	# add to dicts
	dictionary_data={}
	dictionary_labels={}
	dictionary_data['train']=X_train.astype(np.float32)
	dictionary_data['test']=X_test.astype(np.float32)
	dictionary_data['valid']=X_validation.astype(np.float32)
	dictionary_labels['train']=y_train.astype(np.int32)
	dictionary_labels['test']=y_test.astype(np.int32)
	dictionary_labels['valid']=y_validation.astype(np.int32)
	
	return dictionary_data,dictionary_labels

### Metodo per il caricamento di cifar10

def _cifar10_load_data():
	(X_train, y_train), (X_VT, y_VT) = tf.keras.datasets.cifar10.load_data()
	num_classes = 10
	return _prepare_dataset_data(num_classes,X_train,y_train,X_VT,y_VT)
	
### Metodo per il caricamento di mnist-fashion

def _fashion_mnist_load_data():
	(X_train, y_train), (X_VT, y_VT) = tf.keras.datasets.fashion_mnist.load_data()
	
	num_classes = 10
	
	return _prepare_dataset_data(num_classes,X_train,y_train,X_VT,y_VT)

### Metodo per il caricamento di cifar100

def _cifar100_load_data(label_mode = "fine"):
	(X_train, y_train), (X_VT, y_VT) = tf.keras.datasets.cifar100.load_data(label_mode=label_mode)
	num_classes = 0
	if label_mode is "fine":
		num_classes = 100
	elif label_mode is "coarse":
		num_classes = 20
	else:
		return "error"
	return _prepare_dataset_data(num_classes,X_train,y_train,X_VT,y_VT)

###
def _mnist_load_data():
	(X_train, y_train), (X_VT, y_VT) = tf.keras.datasets.mnist.load_data()
	
	num_classes = 10
	
	return _prepare_dataset_data(num_classes,X_train,y_train,X_VT,y_VT)



### Metodi per importare i dati di Marzella


def _parents_read_images(img_size=64, channels=3):
	main_directory_path = "KinFaceW-II"+os.sep+"images"+os.sep
	#father-daughter
	fd_path = main_directory_path+"father-dau"
	fd_prefix = "fd"
	fd_pairs={
		"parent": [],
		"child": []}
	
	#father-son
	fs_path = main_directory_path+"father-son"
	fs_prefix = "fs"
	fs_pairs={
		"parent": [],
		"child": []}
	
	#mother-daughter
	md_path = main_directory_path+"mother-dau"
	md_prefix = "md"
	md_pairs={
		"parent": [],
		"child": []}
	
	#mother-son
	ms_path = main_directory_path+"mother-son"
	ms_prefix = "ms"
	ms_pairs={
		"parent": [],
		"child": []}
	
	data = {
		"paths": [fd_path,fs_path,md_path,ms_path],
		"prefixes": [fd_prefix,fs_prefix,md_prefix,ms_prefix],
		"pairs":[fd_pairs,fs_pairs,md_pairs,ms_pairs]
	}
	
	for i in range(4):
		print("Reading images in: ", data["paths"][i])
		for j in range(250):
			parent_img_path = data["paths"][i]
			parent_img_number = str(j+1).zfill(3)
			parent_img_filename = data["prefixes"][i]+"_"+parent_img_number+"_1.jpg"
			parent_img_path += os.sep+parent_img_filename
			parent_img = cv2.imread(parent_img_path)
			#parent_img = cv2.cvtColor(parent_img, cv2.COLOR_BGR2GRAY)
			parent_img = cv2.resize(parent_img,(img_size,img_size))
			parent_img = np.reshape(parent_img, [img_size,img_size,channels])
			
			child_img_path = data["paths"][i]
			child_img_number = str(j+1).zfill(3)
			child_img_filename = data["prefixes"][i]+"_"+child_img_number+"_2.jpg"
			child_img_path += os.sep+child_img_filename
			child_img = cv2.imread(child_img_path)
			#child_img = cv2.cvtColor(child_img, cv2.COLOR_BGR2GRAY)
			child_img = cv2.resize(child_img,(img_size,img_size))
			child_img = np.reshape(child_img, [img_size,img_size,channels])
			
			data["pairs"][i]["parent"].append(parent_img)
			data["pairs"][i]["child"].append(child_img)
	
	pairs = np.empty(shape=(2000,64,64,6), dtype=np.float32)
	labels = np.empty(shape=(2000), dtype=np.int32)
	
	for i in range(4):
		#positive pairs
		print("Creating positive pairs for: ", data["prefixes"][i])
		for j in range(250):
			parent_img = data["pairs"][i]["parent"][j]
			child_img = data["pairs"][i]["child"][j]
			merged_image = np.concatenate((parent_img,child_img),axis=2)
			pairs[(250*i)+j]=merged_image.astype(np.float32)
			labels[(250*i)+j+250]=i
		
		#negative pairs
		print("Creating negative pairs for: ", data["prefixes"][i])
		for j in range(250):
			parent_img = data["pairs"][i]["parent"][j]
			if j is 249:
				child_img = data["pairs"][i]["child"][0]
			else:
				child_img = data["pairs"][i]["child"][j+1]
			merged_image = np.concatenate((parent_img,child_img),axis=2)
			pairs[(250*i)+j+250]=merged_image.astype(np.float32)
			labels[(250*i)+j+250]=4
	
	print("pairs shape:\n",pairs.shape)
	
	#sec = input('Let us wait for user input. Let me know how many seconds to sleep now.\n')
	
	
	X_train, X_VT, y_train, y_VT = ms.train_test_split(pairs,labels,test_size=0.3, random_state=1)
	X_validation, X_test, y_validation, y_test = ms.train_test_split(X_VT,y_VT,test_size=0.3)
	
	dictionary_data={}
	dictionary_labels={}
	dictionary_data['train']=X_train
	dictionary_data['test']=X_test
	dictionary_data['valid']=X_validation
	dictionary_labels['train']=y_train
	dictionary_labels['test']=y_test
	dictionary_labels['valid']=y_validation
	return dictionary_data,dictionary_labels
	

#### Metodi per l'importazione dei dati del progetto di tesi di Luca Marzella

def _parents_halving_arrays(array):
	new_array = []
	for i in range(len(array)):
		if (i % 2 == 0):
			new_array.append(array[i])
	return new_array

def _parents_read_file(file_name):
	feat_lst = []
	feat_parent = []
	feat_children = []
	class_lst = []
	pairs_lst = []
	i = 0
	with open(file_name) as fr:
		reader = csv.reader(fr, delimiter=',')
		for row in reader:
			clas = int(float(row[-1]))
		  
			row = row[:-1]
			s_feat = [float(i) for i in row]
			s_feat = s_feat
			if (i % 2 == 0):
				feat_parent.append(s_feat)
			else:
				feat_children.append(s_feat)
			feat_lst.append(s_feat)

			class_lst.append(clas)
		 
			i = i + 1
	return feat_lst, class_lst, feat_parent, feat_children


def parents_get_data(pathTrain="TrainSet.txt",pathTest="TestSet.txt", data_cap=600):
	
	##train set fisso contiene anche le coppie spaiate, train set normale invece non le contiene
	_,labels,fP,fC=_parents_read_file(pathTrain)
	labels=_parents_halving_arrays(labels)



	##test set fisso contiene anche le coppie spaiate, test set normale invece non le contiene
	_,test_label,test_fP,test_fC=_parents_read_file(pathTest)
	test_label=_parents_halving_arrays(test_label)

	datasetP=np.array(fP+test_fP)
	datasetC=np.array(fC+test_fC)
	datasetL=np.array(labels+test_label)

	all_combinations=np.empty(shape=(data_cap*2,32,32,1))
	all_labels=np.empty(shape=(data_cap*2))
	#z=0
	for i in range(data_cap):
		boh=np.concatenate((datasetP[i].reshape(512,1),datasetC[i].reshape(512,1)),axis=1)
		boh=boh.reshape(512,2,1)
		boh=boh.reshape(32,32,1)
		all_combinations[i]=boh
		all_labels[i] = datasetL[i]
	for i in range(data_cap):
		indices = random.sample(range(0,data_cap),2)
		boh=np.concatenate((datasetP[indices[0]].reshape(512,1),datasetC[indices[1]].reshape(512,1)),axis=1)
		boh=boh.reshape(512,2,1)
		boh=boh.reshape(32,32,1)
		all_combinations[data_cap+i]=boh
		all_labels[data_cap+i]=4

	X_train, X_VT, y_train, y_VT= ms.train_test_split(all_combinations, all_labels, test_size=0.3, random_state=1)
	X_test, X_validation, y_test, y_validation= ms.train_test_split(X_VT, y_VT, test_size=0.3, random_state=1)
	
	dictionary_data={}
	dictionary_labels={}
	dictionary_data['train']=X_train.astype(np.float32)
	dictionary_data['test']=X_test.astype(np.float32)
	dictionary_data['valid']=X_validation.astype(np.float32)
	dictionary_labels['train']=y_train.astype(np.int32)
	dictionary_labels['test']=y_test.astype(np.int32)
	dictionary_labels['valid']=y_validation.astype(np.int32)
	return dictionary_data,dictionary_labels



