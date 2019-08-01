# -*- coding: utf-8 -*-
import os
import glob
import numpy as np
import pandas as pd

from keras import utils
from keras.models import load_model
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

#load_fire_dataset
from utils.train_model import datasets
from utils.train_model import save_dataframe

root_path = '.'
print('Root path in', root_path)

models_root = os.path.join(root_path, 'models')

# available dataset id: 'fismo', 'firenet', 'cairfire', 'firesense'
train_id = 'firenet' # dataset to test
val_datasets = ['fismo', 'firenet', 'cairfire', 'firesense']
test_datasets = ['fismo', 'firenet_test', 'cairfire', 'firesense'] #'self' for test with same dataset
DEBUG = False
#params redimension
WIDTH = 96
HEIGHT = 96
# params
resize = (WIDTH, HEIGHT)
num_classes = 2
# models path
model_type = 'OctFiResNet'
models_path = '{}_models_{}'.format(model_type, train_id)
save_dir = os.path.join(models_root, models_path)
summary_name = 'testing_summary.csv'

# data storage
data = []
for val_id in val_datasets:
    if train_id == val_id:
        val_id = 'self'
    # model location
    datasets_id = '%s_%s' % (train_id, val_id)
    model_prefix = 'model_{}_*'.format(datasets_id)
    models_path = os.path.join(save_dir, model_prefix)

    for test_id in test_datasets:
        if test_id == train_id or test_id == val_id:
            continue
        # load data
        dt_path = datasets[test_id]['path']
        dt_name = datasets[test_id]['name']
        dt_load = datasets[test_id]['dataset']

        print(dt_name, 'dataset loading..')
        x_test, y_test = dt_load(dt_path, val_split=False,
                            width_resize=WIDTH,
                            height_resize=HEIGHT,
                            debug=DEBUG)
        # Normalize data
        x_test = x_test.astype('float32') / 255
        # sumary
        print('x_test shape:', x_test.shape)
        print(x_test.shape[0], 'test samples')
        print(y_test[y_test==1].shape[0], 'fire')
        print(y_test[y_test==0].shape[0], 'no_fire')

        # Convert class vectors to binary class matrices.
        y_test = utils.to_categorical(y_test, num_classes)

        # evaluate models
        models = glob.glob(models_path)
        for model_path in models:
            fila = []
            model_name = model_path.split(os.path.sep)[-1]
            print('Evaluating', model_name)
            fila.append(model_name)
            fila.append(datasets_id)
            fila.append(test_id)
            # load model
            model = load_model(model_path)

            #Confusion Matrix and Classification Report
            y_pred = model.predict(x_test, verbose=0)
            y_pred = np.argmax(y_pred, axis=1)
            Y_test = np.argmax(y_test, axis=1)

            confusion = confusion_matrix(Y_test, y_pred)
            accuracy = (confusion[0][0] + confusion[1][1]) / np.sum(confusion)
            fila.append(accuracy)
            fila.extend(confusion.flatten())

            target_names = ['No Fire', 'Fire']
            class_report = classification_report(Y_test, y_pred,
                                    target_names=target_names, output_dict=True)

            for key, value in class_report['No Fire'].items():
                fila.append(value)
            for key, value in class_report['Fire'].items():
                fila.append(value)

            data.append(fila)

            col_names = ['model_name', 'datasets_id', 'test_id', 'accuracy',
                         'nofire_as_nofire', 'fire_as_nofire',
                         'nofire_as_fire', 'fire_as_fire',
                         'nofire_precision', 'nofire_recall',
                         'nofire_f1-score', 'nofire_support',
                         'fire_precision', 'fire_recall',
                         'fire_f1-score','fire_support']
            summary = pd.DataFrame(data=data, columns=col_names)
            save_dataframe(summary, save_dir, filename=summary_name)
        # end for
    # end for
# end for
