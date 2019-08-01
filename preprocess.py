# preprocess
import os
from utils.prepare_dataset import prepare_firesense_df
from utils.prepare_dataset import prepare_cairfire_df
from utils.prepare_dataset import prepare_firenet_df
from utils.prepare_dataset import prepare_firenet_test_df
from utils.prepare_dataset import prepare_fismo_df

root_path = os.getcwd()
print('Root path in', root_path)

datasets_root = os.path.join(root_path, 'datasets')

cairfire_dt = os.path.join(datasets_root, 'CairDataset')
firesense_dt = os.path.join(datasets_root, 'FireSenseDataset')
firenet_dt = os.path.join(datasets_root, 'FireNetDataset')
fismo_dt = os.path.join(datasets_root, 'FiSmo-Images')

prepare_firesense_df(firesense_dt)
prepare_cairfire_df(cairfire_dt)
prepare_firenet_df(firenet_dt)
prepare_firenet_test_df(firenet_dt)
prepare_fismo_df(fismo_dt)
