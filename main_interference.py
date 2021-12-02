import os

#os.environ["CUDA_VISIBLE_DEVICES"]= '0'

import h5py
import numpy as np
#import nibabel as nib
import torch
'''%matplotlib inline
%matplotlib notebook
%matplotlib ipympl'''
import matplotlib.pyplot as plt
import time 
#from B0_predict_funtions import HammingFilter_3D, read_vol_data, add_colorbar, train, test, read_and_investigate_TM

from B0_predict_networks import NN_model_modular
import nni

import math
import torch.nn.functional as F

import json

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('device is: ', device)

def test_lol(model, data_in, mini_batch=10):
    model.eval()
    #pred = torch.tensor(np.zeros((data_in.shape[0],1,data_in.shape[2],data_in.shape[3],data_in.shape[4])), dtype=torch.float32)
    pred = np.zeros((data_in.shape[0],1,data_in.shape[2],data_in.shape[3],data_in.shape[4]))
    for i in np.arange(0, data_in.shape[0], mini_batch):
        inn = data_in[i:i+mini_batch, ]
        ti_in = time.time()
        out = model(inn.to(device))
        ti_out = time.time() - ti_in
        print(ti_out)
        pred[i:i+mini_batch, ] = out.cpu().detach().numpy()
        #pred[i:i+mini_batch, ] = out

    return pred


# %% loading model

#hand_ID = 'A49zv'
hand_ID = 'STANDALONE'
epoch = 1500
dir_main = 'Output_data_nni_ver6'

aug_params = json.load(open('{:s}/from_main_{:s}_hyperparameters.json'.format(dir_main, hand_ID)))

model = NN_model_modular(int(aug_params['N_depth']),
                         int(aug_params['N_hidden_feature']),
                         int(aug_params['Kernel_size']),
                         int(aug_params['N_layer'])).to(device)

model.load_state_dict(torch.load('{:s}/from_main_{:s}_after_{:d}_epoch_saved_model.pt'.format(dir_main, hand_ID, epoch)))
#model.load_state_dict(torch.load('Output_data_nni_ver2/from_main_{:s}_saved_model.pt'.format(hand_ID)))

print(model)



# %% loading data
'''
pth = 'Training_data/testing_data_ver_1_0.h5'

pth_s = '{:s}/from_main_{:s}_after_{:d}_test_data.h5'.format(dir_main, hand_ID, epoch)
#pth_s = 'Output_data_nni_ver2/from_main_{:s}_test_data.h5'.format(hand_ID)


fh = h5py.File(pth, 'r')

N = 125

sta = time.time()
input_data_test = torch.zeros((N,3,64,64,64), dtype=torch.float32)
input_data_test[:N,:,0:60,:,:] = torch.tensor(fh['input_data'][:N,:,:60,:64,:64])
print('Loading of input took: {:2.4f}'.format(time.time()-sta))
sta = time.time()

pred_data_test = test_lol(model, input_data_test)
print('Inference took: {:2.4f}'.format(time.time()-sta))
sta = time.time()
fh_s = h5py.File(pth_s, 'a')
fh_s.create_dataset('PR_data_test', data=pred_data_test)
print('Saving of prediction took: {:2.4f}'.format(time.time()-sta))

fh_s.create_dataset('input_data_test', data=input_data_test)
del pred_data_test, input_data_test

sta = time.time()

output_data_test = torch.zeros((N,1,64,64,64), dtype=torch.float32)
output_data_test[:N,:,0:60,:,:] = torch.tensor(fh['output_data'][:N,:,:60,:64,:64])

fh_s.create_dataset('GT_data_test', data=output_data_test)

del output_data_test


coreg_data_test = torch.zeros((N,1,64,64,64), dtype=torch.float32)
coreg_data_test[:N,:,0:60,:,:] = torch.tensor(fh['coreg_data'][:N,:,:60,:64,:64])

fh_s.create_dataset('PMS_data_test', data=coreg_data_test)

del coreg_data_test


coreg_Mask_test = torch.zeros((N,1,64,64,64), dtype=torch.float32)
coreg_Mask_test[:N,:,0:60,:,:] = torch.tensor(fh['coreg_Mask'][:N,:,:60,:64,:64])

fh_s.create_dataset('mask_data_test', data=coreg_Mask_test)

del coreg_Mask_test

print('The rest took: {:2.4f}'.format(time.time()-sta))

fh.close()
fh_s.close()'''

# %% Loading and saving init mask
'''pth = 'Training_data/testing_data_ver_1_0_only_init_mask.h5'
pth_s = '{:s}/from_main_{:s}_after_{:d}_test_data_only_init_mask.h5'.format(dir_main, hand_ID, epoch)
N=5
fh = h5py.File(pth, 'r')
init_Mask_test = torch.zeros((N,1,64,64,64), dtype=torch.float32)
init_Mask_test[:N,:,0:60,:,:] = torch.tensor(fh['init_mask_test'][:N,:,:60,:64,:64])
fh.close()

fh_s = h5py.File(pth_s, 'a')
fh_s.create_dataset('init_Mask_test', data=init_Mask_test)
fh_s.close()'''


# %% Loading training data 

'''pth = 'Training_data/training_data_ver_0_2.h5'
pth_s = '{:s}/from_main_{:s}_after_{:d}_train_data.h5'.format(dir_main, hand_ID, epoch)
fh_s = h5py.File(pth_s, 'a')

h = h5py.File(pth, 'r')

N = 950
sta = time.time()
input_data = torch.zeros((N,3,64,64,64), dtype=torch.float32)
input_data[:, :, 0:60, :, :] = torch.tensor(h['input_data'][:N, :3, :60, :64, :64], dtype=torch.float32)
print('Loading of input train took: {:2.4f}'.format(time.time()-sta))    


sta = time.time()
pred_data_train = test_lol(model, input_data)
print('Inference took: {:2.4f}'.format(time.time()-sta))
sta = time.time()


fh_s.create_dataset('PR_data_train', data=pred_data_train)


output_data = torch.zeros((N,1,64,64,64), dtype=torch.float32)
output_data[:, :, 0:60, :, :] = torch.tensor(h['output_data'][:N, :1, :60, :64, :64], dtype=torch.float32)
fh_s.create_dataset('GT_data_train',data=output_data)
del output_data

coreg_Mask = torch.zeros((N,1,64,64,64), dtype=torch.float32)
coreg_Mask[:, :, 0:60, :, :] = torch.tensor(h['coreg_Mask'][:N, :1, :60, :64, :64], dtype=torch.float32)
fh_s.create_dataset('mask_data_train',data=coreg_Mask)
del coreg_Mask

coreg_data = torch.zeros((N,1,64,64,64), dtype=torch.float32)
coreg_data[:N,:,0:60,:,:] = torch.tensor(h['coreg_data'][:N,:,:60,:64,:64])
fh_s.create_dataset('PMS_data_train',data=coreg_data)
del coreg_data

h.close()
fh_s.close()
'''


# %% Loading precision data 


pth = 'Training_data/precision_data_ver_1_0.h5'
pth_s = '{:s}/from_main_{:s}_after_{:d}_precision_data_2.h5'.format(dir_main, hand_ID, epoch)

fh = h5py.File(pth, 'r')
fh_s = h5py.File(pth_s, 'a')


#N = 200
N = 50
input_data = torch.zeros((N,3,64,64,64), dtype=torch.float32)
input_data[:, :, 0:60, :, :] = torch.tensor(fh['input_data'][:N, :3, :60, :64, :64], dtype=torch.float32)

pred_data_precision = test_lol(model, input_data, mini_batch=1)

exit()

fh_s.create_dataset('PR_data_prec', data=pred_data_precision)
fh_s.create_dataset('input_data_prec', data=input_data)

del pred_data_precision, input_data


output_data = torch.zeros((N,1,64,64,64), dtype=torch.float32)
output_data[:N,:,0:60,:,:] = torch.tensor(fh['output_data'][:N,:,:60,:64,:64])

fh_s.create_dataset('GT_data_prec', data=output_data)

del output_data

coreg_Mask_test = torch.zeros((N,1,64,64,64), dtype=torch.float32)
coreg_Mask_test[:N,:,0:60,:,:] = torch.tensor(fh['coreg_Mask'][:N,:,:60,:64,:64])

fh_s.create_dataset('mask_data_prec', data=coreg_Mask_test)

del coreg_Mask_test

fh.close()
fh_s.close()


