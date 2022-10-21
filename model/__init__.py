from importlib import import_module

import torch
import torch.nn as nn
import torch.nn.functional as F
import model.common

class model:
    def __init__(self, opt):
        self.module = import_module('model.' + opt.model)
        self.opt = opt

    def get_model(self):
        print('Making model...')
        module = import_module('model.' + self.opt.model)
        my_model = module.make_model()

        # continue training from previous state
        if self.opt.pre_train != '.':
            print('Loading model from {}...'.format(self.opt.pre_train))
            # my_model.load_state_dict(torch.load(self.opt.pre_train))
        # else:
        #     my_model.apply(common.weights_init_vdsr)
            
        # Enable cuda
        if not self.opt.no_cuda:
            print('\tCUDA is ready!')
            torch.cuda.manual_seed(self.opt.seed)
            my_model.cuda()
            if self.opt.n_GPUs > 1:
                my_model = nn.DataParallel(my_model, range(0, self.opt.n_GPUs))

        
        
        common.print_network(my_model)

        return my_model

