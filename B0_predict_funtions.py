import numpy as np
import nibabel as nib
from mpl_toolkits import axes_grid1
import matplotlib.pyplot as plt
import torch

# %% Hamming filtering in all three spatial dimension

# = T1w_RG #  [0,:,:,:]

def HammingFilter_3D(m,n_x=64,n_z=60):
    DV = (1,2,3)

    HF = np.tile(np.hamming(n_x), (n_x,1))
    HF = np.tile(np.expand_dims(HF * HF.transpose(),0), (n_z, 1, 1))
    HF60 = np.tile(np.expand_dims(np.hamming(n_z), (1,2)), (1, n_x, n_x))

    HF_ff = np.tile(np.expand_dims(HF * HF60, 0), (m.shape[0],1,1,1))


    '''m = np.fft.fft(np.fft.fft(m, axis=DV[0]), axis=DV[1])
    print(DV)'''
    M = np.fft.fftshift(np.fft.fftshift(np.fft.fftshift( np.fft.fft(np.fft.fft(np.fft.fft(m, axis=DV[0]), axis=DV[1]), axis=DV[2]) ,axes=DV[0]),axes=DV[1]),axes=DV[2])
    M = M * HF_ff
    mr = np.fft.ifft(np.fft.ifft(np.fft.ifft(np.fft.ifftshift(np.fft.ifftshift(np.fft.ifftshift(M,axes=DV[0]),axes=DV[1]),axes=DV[2]) ,axis=DV[0]),axis=DV[1]),axis=DV[2])

    return mr


def read_vol_data(m_pth, base_name, N=[0,40], HF=False, ResName=''):
    
    if not N:
        Mat = np.zeros((1,60,64,64))
        nii_struct = nib.load('{:s}/{:s}{:s}.nii'.format(m_pth,base_name,ResName))
        Mat[0,] = nii_struct.get_fdata()
        
        
    else:

        Mat = np.zeros((N[1],60,64,64))
        for i in np.arange(N[0],N[1]+1):
            try:
                # print('{:s}/B0_map_{:d}.nii'.format(m_pth,i))
                nii_struct = nib.load('{:s}/{:s}{:d}{:s}.nii'.format(m_pth,base_name,i,ResName))
                Mat[i-1,] = nii_struct.get_fdata()
                
            except:
                print('{:s}/{:s}{:d}{:s}.nii'.format(m_pth,base_name,i,ResName))
                Mat = Mat[0:i-1,]
                break
    
    if HF==True:
        #Mat = HammingFilter_3D(Mat).real
        Mat = (HammingFilter_3D(Mat))
    
    return Mat


def read_and_investigate_TM(pth,bsn,N=[0,40]):
    ## Analysis of rotations ## 
    #pth = 'Vol_016_DG_B0/B0s_final'
    #N = 28

    rot_angle_all = np.zeros((N[1],1))
    rot_axes_all = np.zeros((N[1],3))

    mov_vect_all = np.zeros((N[1],3))

    #norm_vectors = np.zeros((N[1],4))

    for i in np.arange(N[0],N[1]+1):
        try:
            #print('{:s}/{:s}{:d}.mat'.format(pth, bsn, i))
            RM = np.loadtxt('{:s}/{:s}{:d}.mat'.format(pth, bsn, i))

            rot_angle_all[i-1,0] = np.arccos(( RM[0,0] + RM[1,1] + RM[2,2] - 1)/2) * 180 / np.pi

            rot_axes_all[i-1,0] = (RM[2,1] - RM[1,2])/np.sqrt(np.square(RM[2,1] - RM[1,2])+np.square(RM[0,2] - RM[2,0])+np.square(RM[1,0] - RM[0,1]))
            rot_axes_all[i-1,1] = (RM[0,2] - RM[2,0])/np.sqrt(np.square(RM[2,1] - RM[1,2])+np.square(RM[0,2] - RM[2,0])+np.square(RM[1,0] - RM[0,1]))
            rot_axes_all[i-1,2] = (RM[1,0] - RM[0,1])/np.sqrt(np.square(RM[2,1] - RM[1,2])+np.square(RM[0,2] - RM[2,0])+np.square(RM[1,0] - RM[0,1]))

            mov_vect_all[i-1,:] = RM[0:3,3]

            #norm_vectors[i-1, :] = np.squeeze(np.matmul(RM, np.array([[1],[1],[1],[0]])))
        except:
            print('{:s}/{:s}{:d}.mat'.format(pth, bsn, i), 'such doesnt exist')
            
            rot_angle_all = rot_angle_all[N[0]-1:i-1,]
            rot_axes_all = rot_axes_all[N[0]-1:i-1,]
            mov_vect_all = mov_vect_all[N[0]-1:i-1,]
            
            mov_vect_all_mag = np.sqrt(np.sum(mov_vect_all**2, axis=1))
                    
            break

    return rot_angle_all, rot_axes_all, mov_vect_all, mov_vect_all_mag
    

def add_colorbar(im, aspect=20, pad_fraction=0.5, **kwargs):
    
    """Add a vertical color bar to an image plot."""
    divider = axes_grid1.make_axes_locatable(im.axes)
    width = axes_grid1.axes_size.AxesY(im.axes, aspect=1./aspect)
    pad = axes_grid1.axes_size.Fraction(pad_fraction, width)
    current_ax = plt.gca()
    cax = divider.append_axes("right", size=width, pad=pad)
    plt.sca(current_ax)
    
    return im.axes.figure.colorbar(im, cax=cax, **kwargs)


def train(model, optimizer, dat_tr_in, dat_tr_out, mini_batch=2):
    model.train()
    total_loss = 0
    mini_batch_idx = 0
    mini_batch_loss = 0

    for i in np.arange(0, dat_tr_in.shape[0], mini_batch):
        mini_batch_idx +=1
        optimizer.zero_grad()
        inn = dat_tr_in[i:i+mini_batch, ]
        out = model(inn)
        loss = torch.nn.functional.mse_loss(dat_tr_out[i:i+mini_batch, ], out)
        loss.backward()
        optimizer.step()

        total_loss += loss

    return total_loss


def test(model, data_in, mini_batch=10):
    model.eval()
    #pred = torch.tensor(np.zeros((data_in.shape[0],1,data_in.shape[2],data_in.shape[3],data_in.shape[4])))
    pred = np.zeros((data_in.shape[0],1,data_in.shape[2],data_in.shape[3],data_in.shape[4]))
    for i in np.arange(0, data_in.shape[0], mini_batch):
        inn = data_in[i:i+mini_batch, ]
        out = model(inn)
        pred[i:i+mini_batch, ] = out.cpu().detach().numpy()
        #pred[i:i+mini_batch, ] = out
        
    return pred


