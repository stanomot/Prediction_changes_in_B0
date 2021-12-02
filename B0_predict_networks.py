import torch

'''def NN_model(nf=2,ks=5):

    class Net(torch.nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            #nf = 2
            self.conv1 = torch.nn.Conv3d(3, 2*nf, ks, padding=2)
            self.conv1_1 = torch.nn.Conv3d(2*nf, 4*nf, ks, padding=2)
            self.conv1_2 = torch.nn.Conv3d(4*nf, 8*nf, ks, padding=2)

            self.maxpool = torch.nn.MaxPool3d(2)

            self.conv2 = torch.nn.Conv3d(8*nf, 8*nf, ks, padding=2)
            self.conv2_1 = torch.nn.Conv3d(8*nf, 16*nf, ks, padding=2)
            self.conv2_2 = torch.nn.Conv3d(16*nf, 32*nf, ks, padding=2)


            self.conv3 = torch.nn.Conv3d(32*nf, 64*nf, ks, padding=2)
            self.conv3_1 = torch.nn.Conv3d(64*nf, 64*nf, ks, padding=2)
            self.conv3_2 = torch.nn.Conv3d(64*nf, 32*nf, ks, padding=2)

            self.up = torch.nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)

            self.inv_conv2 = torch.nn.ConvTranspose3d(64*nf, 32*nf, ks, padding=2)
            self.inv_conv2_1 = torch.nn.ConvTranspose3d(32*nf, 16*nf, ks, padding=2)
            self.inv_conv2_2 = torch.nn.ConvTranspose3d(16*nf, 8*nf, ks, padding=2)

            self.inv_conv1 = torch.nn.ConvTranspose3d(16*nf, 8*nf, ks, padding=2)
            self.inv_conv1_1 = torch.nn.ConvTranspose3d(8*nf, 4*nf, ks, padding=2)
            self.inv_conv1_2 = torch.nn.ConvTranspose3d(4*nf, 1, ks, padding=2)

        def forward(self, dat):
            dat = torch.tanh(self.conv1(dat))
            dat = torch.tanh(self.conv1_1(dat))
            dat_bef1 = torch.tanh(self.conv1_2(dat))
            dat = self.maxpool(dat_bef1)  # nf 8*8

            dat = torch.tanh(self.conv2(dat))
            dat = torch.tanh(self.conv2_1(dat))
            dat_bef2 = torch.tanh(self.conv2_2(dat))
            dat = self.maxpool(dat_bef2)  # nf 32*8

            dat = torch.tanh(self.conv3(dat))
            dat = torch.tanh(self.conv3_1(dat))
            dat = torch.tanh(self.conv3_2(dat))  # 32*8

            dat = self.up(dat)
            dat = torch.cat((dat_bef2, dat), 1)
            dat = torch.tanh(self.inv_conv2(dat))
            dat = torch.tanh(self.inv_conv2_1(dat))
            dat = torch.tanh(self.inv_conv2_2(dat))  # 8*8

            dat = self.up(dat)
            dat = torch.cat((dat_bef1, dat), 1)
            dat = torch.tanh(self.inv_conv1(dat))
            dat = torch.tanh(self.inv_conv1_1(dat))
            dat = (self.inv_conv1_2(dat))

            return dat
        
    model = Net()
    
    return model 

def NN_model_modular(depth, hf, ks, n_L):
        
    class LvL_down(torch.nn.Module):
        def __init__(self,fi,fh,ks,n_layers=4):
            super(LvL_down, self).__init__()
            layers = [torch.nn.Conv3d(fi,fh, ks, padding=2),
                     torch.nn.Tanh()]
            for i in range (n_layers-1):
                layers += [torch.nn.Conv3d(fh, fh, ks, padding=2),
                          torch.nn.Tanh()]
                
            self.c = torch.nn.Sequential(*layers)
            
        def forward(self,dat):
            
            dat = self.c(dat)

            return dat
        

    class LvL_up(torch.nn.Module):
        def __init__(self,fi, fo, ks, n_layers=4):
            super(LvL_up, self).__init__()
            
            layers = [torch.nn.ConvTranspose3d(fi, fo, ks, padding=2),
                     torch.nn.Tanh()]
            
            for i in range(n_layers-1):
                layers += [torch.nn.ConvTranspose3d(fo, fo, ks, padding=2),
                          torch.nn.Tanh()]
            
            self.c = torch.nn.Sequential(*layers)
            
        def forward(self,dat):
            
            dat = self.c(dat)

            
            return dat
        
        
    class Net(torch.nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            
            self.down_path = torch.nn.ModuleList()
            self.up_path = torch.nn.ModuleList()
            
            self.down_path.append(LvL_down(3,2*hf,ks,n_L)) # 0 lvl
            
            for i in range(1,depth):
                self.down_path.append(LvL_down(2**i*hf,2**(i+1)*hf, ks,n_L))
            
            for i in reversed(range(1,depth)):
                self.up_path.append(LvL_up(2**(i+1)*hf, 2**i*hf, ks,n_L))
                                                                          
        def forward(self,dat):
            
            intermediate_step = list()
            for i, down in enumerate(self.down_path):
                dat = down(dat)
                intermediate_step.append(dat)
                
            return dat
        
        
    model = Net()
    return model'''



def NN_model_modular(depth, hf, ks, n_L):
    class LvL_down(torch.nn.Module):
        def __init__(self, fi, fh, ks, n_layers=4):
            super(LvL_down, self).__init__()
            layers = [torch.nn.Conv3d(fi, fh, ks, padding=int(ks/2)),
                      torch.nn.Tanh()]
            for i in range(n_layers - 1):
                layers += [torch.nn.Conv3d(fh, fh, ks, padding=int(ks/2)),
                           torch.nn.Tanh()]

            self.c = torch.nn.Sequential(*layers)

        def forward(self, dat):
            dat = self.c(dat)

            return dat

    class Depeest_lvl(torch.nn.Module):
        def __init__(self, fi, fh, ks):
            super(Depeest_lvl, self).__init__()
            lvl = [torch.nn.Conv3d(fi, fh, ks, padding=int(ks/2)),
                   torch.nn.Tanh(),
                   torch.nn.Conv3d(fh, fh, ks, padding=int(ks/2)),
                   torch.nn.Tanh(),
                   torch.nn.ConvTranspose3d(fh, fi, ks, padding=int(ks/2)),
                   torch.nn.Tanh()]

            self.bottleneck = torch.nn.Sequential(*lvl)

        def forward(self, dat):
            dat = self.bottleneck(dat)

            return dat

    class LvL_up(torch.nn.Module):
        def __init__(self, fi, fo, ks, n_layers=4):
            super(LvL_up, self).__init__()

            layers = [torch.nn.ConvTranspose3d(2*fi, fi, ks, padding=int(ks/2)),
                      torch.nn.Tanh()]

            for i in range(n_layers - 1):
                layers += [torch.nn.ConvTranspose3d(fi, fi, ks, padding=int(ks/2)),
                           torch.nn.Tanh()]

            layers += [torch.nn.ConvTranspose3d(fi, fo, ks, padding=int(ks/2)),
                       torch.nn.Tanh()]

            self.c = torch.nn.Sequential(*layers)

        def forward(self, dat):
            dat = self.c(dat)

            return dat

    class Net(torch.nn.Module):
        def __init__(self):
            super(Net, self).__init__()

            self.down_path = torch.nn.ModuleList()

            self.down_path.append(LvL_down(3, 2 * hf, ks, n_L))  # 0 lvl

            for i in range(1, depth):
                self.down_path.append(LvL_down(2 ** i * hf, 2 ** (i + 1) * hf, ks, n_L))

            self.bottleneck = Depeest_lvl(2 ** depth * hf, 2 ** (depth+1) * hf, ks)

            self.up_path = torch.nn.ModuleList()

            for i in reversed(range(0, depth)):
                if i != 0:
                    self.up_path.append(LvL_up(2 ** (i + 1) * hf, 2 ** i * hf, ks, n_L-1))
                else:
                    self.up_path.append(LvL_up(2 ** (i + 1) * hf, 1, ks, n_L - 1))
                    
            self.max_pool = torch.nn.MaxPool3d(2)
            self.up_sample = torch.nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)

        def forward(self, dat):

            intermediate_step = list()
            for i, down in enumerate(self.down_path):
                dat_current = down(dat)
                intermediate_step.append(dat_current)
                dat = self.max_pool(dat_current)

            dat = self.bottleneck(dat)

            for i, up in enumerate(self.up_path):
                dat_current = self.up_sample(dat)
                dat = torch.cat((intermediate_step[depth-1-i], dat_current), 1)
                dat = up(dat)
            return dat

    model = Net()
    return model
