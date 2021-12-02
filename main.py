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


def gaussian_kernel(size, sigma=2., dim=2, channels=3):
    # The gaussian kernel is the product of the gaussian function of each dimension.
    # kernel_size should be an odd number.

    kernel_size = 2 * size + 1

    kernel_size = [kernel_size] * dim
    sigma = [sigma] * dim
    kernel = 1
    meshgrids = torch.meshgrid([torch.arange(size, dtype=torch.float32) for size in kernel_size])

    for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
        mean = (size - 1) / 2
        kernel *= 1 / (std * math.sqrt(2 * math.pi)) * torch.exp(-((mgrid - mean) / (2 * std)) ** 2)

    # Make sure sum of values in gaussian kernel equals 1.
    kernel = kernel / torch.sum(kernel)

    # Reshape to depthwise convolutional weight
    kernel = kernel.view(1, 1, *kernel.size())
    kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))

    return kernel


def Gaussian_blur(x, size=2, channels=1, device=device):
    kernel = gaussian_kernel(size=size,dim=3,channels=channels)
    kernel_size = 2 * size + 1

    #x = x[None, ...]
    padding = int((kernel_size - 1) / 2)
    x = F.pad(x, (padding, padding, padding, padding, padding, padding), mode='replicate')
    #x = F.pad(x, padding, mode='replicate')
    x = F.conv3d(x, kernel.to(device), groups=1)

    return x


def test_lol(model, data_in, mini_batch=10):
    model.eval()
    #pred = torch.tensor(np.zeros((data_in.shape[0],1,data_in.shape[2],data_in.shape[3],data_in.shape[4])), dtype=torch.float32)
    pred = np.zeros((data_in.shape[0],1,data_in.shape[2],data_in.shape[3],data_in.shape[4]))
    for i in np.arange(0, data_in.shape[0], mini_batch):
        inn = data_in[i:i+mini_batch, ]
        out = model(inn.to(device))
        pred[i:i+mini_batch, ] = out.cpu().detach().numpy()
        #pred[i:i+mini_batch, ] = out

    return pred

def test_val(model, dat_tr_in, dat_tr_out, mini_batch=10, mask=[], mask_flag=False, weig=0.01):
    model.eval()
    total_loss = 0
    mini_batch_idx = 0
    mini_batch_loss = 0
    
    for i in np.arange(0, dat_tr_in.shape[0], mini_batch):
        mini_batch_idx += 1
        #optimizer.zero_grad()
        inn = dat_tr_in[i:i + mini_batch, ]
        out = model(inn.to(device))
        out_gt = dat_tr_out[i:i + mini_batch, ].to(device)
        # print(out.shape, out_gt.shape, mask[i:i+mini_batch, ].shape)
        # print(mask[i:i+mini_batch, ])
        if not mask_flag:
            loss = torch.nn.functional.mse_loss(out_gt, out)
        else:
            #loss1 = torch.nn.functional.mse_loss(out_gt[mask[i:i + mini_batch, ]], out[mask[i:i + mini_batch, ]])
            loss1 = torch.nn.functional.mse_loss(out_gt, out)
            sm = out - Gaussian_blur(out)
            loss2 = torch.mean(torch.abs(sm[mask[i:i + mini_batch, ]]))
            loss = (1-weig)*loss1 + weig*loss2
    
        total_loss += loss.detach()
        
    return total_loss

def train(model, optimizer, dat_tr_in, dat_tr_out, mini_batch=2, mask=[], mask_flag=False, weig=0.01):
    model.train()
    total_loss = 0
    mini_batch_idx = 0
    mini_batch_loss = 0

    for i in np.arange(0, dat_tr_in.shape[0], mini_batch):
        mini_batch_idx += 1
        optimizer.zero_grad()
        #inn = dat_tr_in[i:i + mini_batch, ]
        out = model(dat_tr_in[i:i + mini_batch, ].to(device))
        out_gt = dat_tr_out[i:i + mini_batch, ]
        
        
        
        # print(out.shape, out_gt.shape, mask[i:i+mini_batch, ].shape)
        # print(mask[i:i+mini_batch, ])
        if not mask_flag:
            loss = torch.nn.functional.mse_loss(out_gt.to(device), out)
            del out, out_gt
        else:
            #loss1 = torch.nn.functional.mse_loss(out_gt[mask[i:i + mini_batch, ]], out[mask[i:i + mini_batch, ]])
            loss1 = torch.nn.functional.mse_loss(out_gt.to(device), out)
            sm = out - Gaussian_blur(out)
            loss2 = torch.mean(torch.abs(sm[mask[i:i + mini_batch, ]]))
            loss = (1-weig)*loss1 + weig*loss2
            
            del out, out_gt, loss1, loss2, sm

        loss.backward()
        optimizer.step()

        total_loss += loss.detach()
        del loss
    
    return total_loss
    
    
def main(aug_params):
    h = h5py.File('Training_data/training_data_ver_0_2.h5','r')

    N_all = 950
    N = 900
    
    idx_load = np.random.permutation(N_all)
    idx_load = idx_load[:N]
    idx_load.sort()
    
    print(idx_load)
    
    start = time.time()
    
    input_data = torch.zeros((N,3,64,64,64), dtype=torch.float32)
    input_data[:, :, 0:60, :, :] = torch.tensor(h['input_data'][idx_load, :3, :60, :64, :64], dtype=torch.float32)
    
    print(input_data.shape)
    
    prem_idx = torch.randperm(input_data.shape[0])
    T_idx = int(0.80*N)
    
    input_data_tr = input_data[prem_idx[0:T_idx], ]
    input_data_vl = input_data[prem_idx[T_idx: ], ]
    
    del input_data
    
    output_data = torch.zeros((N,1,64,64,64), dtype=torch.float32)
    output_data[:, :, 0:60, :, :] = torch.tensor(h['output_data'][idx_load, :1, :60, :64, :64], dtype=torch.float32)
    output_data_tr = output_data[prem_idx[0:T_idx], ]
    output_data_vl = output_data[prem_idx[T_idx: ], ]
    
    del output_data
    
    coreg_Mask = torch.zeros((N,1,64,64,64), dtype=torch.float32)
    coreg_Mask[:, :, 0:60, :, :] = torch.tensor(h['coreg_Mask'][idx_load, :1, :60, :64, :64], dtype=torch.float32)
    coreg_Mask_tr = coreg_Mask[prem_idx[0:T_idx], ]
    coreg_Mask_vl = coreg_Mask[prem_idx[T_idx: ], ]
    
    del coreg_Mask
    
    coreg_Mask_bool_tr = torch.tensor(coreg_Mask_tr >0, dtype=torch.bool).to(device)
    coreg_Mask_bool_vl = torch.tensor(coreg_Mask_vl >0, dtype=torch.bool).to(device)
    
    
    coreg_B0_data = torch.zeros((N,1,64,64,64), dtype=torch.float32)
    coreg_B0_data[:,:, 0:60, :, :] = torch.tensor(h['coreg_data'][idx_load, :1, :60, :64, :64], dtype=torch.float32)
    
    coreg_B0_data_tr = coreg_B0_data[prem_idx[0:T_idx], ]
    coreg_B0_data_vl = coreg_B0_data[prem_idx[T_idx:], ]
    
    del coreg_B0_data
    

    
    print('took: ',time.time()-start)

    # asking for model to load

    model = NN_model_modular(int(aug_params['N_depth']),
                             int(aug_params['N_hidden_feature']),
                             int(aug_params['Kernel_size']),
                             int(aug_params['N_layer'])).to(device)
    
    opt = torch.optim.Adam(model.parameters(), lr=aug_params['learning_rate'])
    
    print(model)

    exit()
    # load model weights checkpoint
    '''hand_ID = 'STANDALONE'
    epoch = 700
    dir_main = 'Output_data_nni_ver5'
    model.load_state_dict(torch.load('{:s}/from_main_{:s}_after_{:d}_epoch_saved_model.pt'.format(dir_main, hand_ID, epoch)))'''
    
    print('carring on')
    
    N_epoch_1st = 1
    N_epoch = 1501
    
    weig = aug_params['l_regularizer']

    # the lowest loss ever 0.00546521
    #pred_vl_1case = np.zeros((N_epoch,52,64))

    
    json.dump(aug_params , open('Output_data_nni_ver7/from_main_{:s}_hyperparameters.json'.format(nni.trial.get_trial_id()), 'w'))

    for epoch in range(N_epoch_1st,N_epoch):
        
        #cur_loss = train(model, opt, input_data[0:28,], output_data[0:28,], mini_batch=2, mask=coreg_Mask_bool[0:28,])
        #cur_loss = train(model, opt, input_data[0:28,], output_data[0:28,], mini_batch=2)
        
        sta = time.time()
        cur_loss = train(model, opt, input_data_tr, output_data_tr, mini_batch=int(aug_params['mini_batch']), mask=coreg_Mask_bool_tr, mask_flag=True, weig=weig)
        sto = time.time()-sta
        
        sta = time.time()
        val_loss = test_val(model, input_data_vl, output_data_vl, mask_flag=True, mask=coreg_Mask_bool_vl, weig=weig)
        sto_vl = time.time()-sta
        

        
        #print(type(cur_loss.cpu().detach().numpy()))
        #print(cur_loss.cpu().detach().numpy())
        #nni.report_intermediate_result(cur_loss.cpu().detach().numpy())
        nni.report_intermediate_result(val_loss.item())
        #print(cur_loss.item())
        
        #sta_vl = time.time()
        #pred_vl = test(data_in_vl)
        #loss_vl = torch.nn.functional.mse_loss(data_out_vl.cpu(), pred_vl.cpu())

        #pred_vl_1case[epoch, ] = pred_vl[6, 0, 33, :, :].cpu().detach().numpy()
        #sto_vl = time.time()-sta_vl

        #loss_vl = 0
        #sto_vl = 0

        log = 'Epoch: {:03d}, Train: {:.8f}, Time: {:.4f}, Valid: {:.8f}, Time_valid: {:.4f}'
        print(log.format(epoch, cur_loss, sto, val_loss, sto_vl))
        
        del cur_loss
        
        if not epoch % 50:
            print('Saving intermediate results of after {:d} epoch'.format(epoch))
            torch.save(model.state_dict(), 'Output_data_nni_ver7/from_main_{:s}_after_{:d}_epoch_saved_model.pt'.format(nni.trial.get_trial_id(), epoch))
            

        
    #nni.report_final_result(cur_loss.cpu().detach().numpy())
    nni.report_final_result(val_loss.item())
    '''
    print('')
    print('')
    print('')
    
    
    print('Training finished let s predict')
    print('')
    print('')
    print('')
    
    data_pred_vl = test_lol(model, input_data_vl)
    

    print('Let s predict save')
    print('')
    print('')
    print('')
    
    
    pth = 'Output_data_nni_ver5/from_main_{:s}.h5'.format(nni.trial.get_trial_id())

    fh = h5py.File(pth, 'a')
    fh.create_dataset('coreg_masks_test', data=coreg_Mask_vl[:,0,:,:,:].numpy())
    #fh.create_dataset('coreg_dat_test', data=coreg_B0_data_test[:,0,:,:,:])
    fh.create_dataset('GT_dat_test', data=output_data_vl[:,0,:,:,:].numpy())
    fh.create_dataset('pred_dat_test', data=data_pred_vl[:,0,:,:,:])
    fh.create_dataset('coreg_data_test', data=coreg_B0_data_vl[:,0,:,:,:].numpy())
    
    fh.close()
    
    
    
    torch.save(model.state_dict(), 'Output_data_nni_ver5/from_main_{:s}_saved_model.pt'.format(nni.trial.get_trial_id()))'''
    
    

if __name__ == '__main__':
    aug_params = {'learning_rate': 0.000025,
                 'mini_batch': 12,
                 'N_depth': 3,
                 'N_hidden_feature': 4,
                 'Kernel_size': 5,
                 'N_layer': 3,
                 'l_regularizer': 0.001}
       
    #aug_params = nni.get_next_parameter()
    main(aug_params)