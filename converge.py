import h5py
import numpy as np
h=h5py.File('yolo.h5','w')
h['data']=np.load('.\model_data\darknet19_448.conv.23',allow_pickle=True)
h['config']=np.load('.\model_data\yolov2.cfg',allow_pickle=True)
h.close()
