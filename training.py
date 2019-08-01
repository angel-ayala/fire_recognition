# -*- coding: utf-8 -*-
import os
import matplotlib.pyplot as plt

#load_fire_dataset
from utils.train_model import train_model
from utils.train_model import summarize_model
from utils.train_model import datasets

# root_path = os.path.join(drive_root, 'My Drive') # os.getcwd()
root_path = ''
print('Root path in', root_path)

models_root = os.path.join(root_path, 'models')
# available dataset id: 'fismo', 'firenet', 'cairfire', 'firesense'
train_id = 'firenet' # dataset to train
val_datasets = ['firenet', 'fismo', 'cairfire', 'firesense']

#params redimension
WIDTH = 96
HEIGHT = 96
# params
DEBUG = False # shows paths of images on loading and mor
input_shape = (WIDTH, HEIGHT, 3)
learning_rate = 1e-4
epochs = 100
batch_size = 32
iterations = 10 # number of repeat the training
iter_start = 0
model_type = 'OctFiResNet' # name proposed model
folder_name = '{}_models_{}'.format(model_type, train_id)
save_dir = os.path.join(models_root, folder_name)
make_summary = False

if make_summary:
    # model.summary() and plot_model()
    summarize_model(save_dir, model_type)

# train against other dataset
for val_id in val_datasets:
    if train_id == val_id:
        dt_name = 'itself'
        val_id = None
    else:
        dt_name = datasets[val_id]['name']
    print('Training', datasets[train_id]['name'],'dataset against', dt_name)
    train_model(iterations=iterations,
                train_dataset=train_id,
                iter_start=iter_start,
                test_dataset=val_id,
                save_dir=save_dir,
                batch_size=batch_size,
                epochs=epochs,
                input_shape=input_shape,
                debug=DEBUG)
