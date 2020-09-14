# Improved-ENAS
An improvement upon the tensorflow implementation (https://github.com/MINGUKKANG/ENAS-Tensorflow) of Efficient Neural Architecture Search (Pham et al, 2018), which focuses on adding a system to modify the reward used during the LSTM controller training phase.

## How to

### Setup

You should modify the following code in main_controller_child_trainer.py to suit your specific situation:

```python
DEFINE_string("output_dir", "."+os.path.sep+"output" , "")
DEFINE_string("child_log_filename","child_log.csv","")
DEFINE_string("controller_log_filename","controller_log.csv","")
DEFINE_string("architecture_info_filename","arc_info.json","")
DEFINE_string("best_arcs_filename","best_arcs.csv","")
DEFINE_string("train_data_dir", "."+os.path.sep+"data"+os.path.sep+"train", "")
DEFINE_string("val_data_dir", "."+os.path.sep+"data"+os.path.sep+"valid", "")
DEFINE_string("test_data_dir", "."+os.path.sep+"data"+os.path.sep+"test", "")
DEFINE_integer("channel",3, "MNIST/fashion_MNIST: 1, Cifar10/Cifar100: 3, parents: 3, parents_img: 6")
DEFINE_integer("img_size", 32, "enlarge image size")
DEFINE_integer("n_aug_img",1 , "if 2: num_img: 55000 -> aug_img: 110000, elif 1: False")
DEFINE_string("data_source","cifar10","either 'mnist', 'cifar10', 'fashion_mnist' or anything else to load a custom dataset ")
```

If custom training data is to be supplied

### Execution
