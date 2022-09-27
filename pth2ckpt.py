# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""
python pth2ckpt.py
"""
import torch
from mindspore.train.serialization import save_checkpoint
from mindspore import Tensor
import mindspore as ms

param = {
    'bn1.bias': 'bn1.beta',
    'bn1.weight': 'bn1.gamma',
    'IN.weight': 'IN.gamma',
    'IN.bias': 'IN.beta',
    'BN.bias': 'BN.beta',
    'in.weight': 'in.gamma',
    'bn.weight': 'bn.gamma',  # *****
    'bn.bias': 'bn.beta',  # *****
    'bn2.weight': 'bn2.gamma',
    'bn2.bias': 'bn2.beta',
    'bn3.bias': 'bn3.beta',
    'bn3.weight': 'bn3.gamma',
    'BN.running_mean': 'BN.moving_mean',
    'BN.running_var': 'BN.moving_variance',
    'bn.running_mean': 'bn.moving_mean',   # *****
    'bn.running_var': 'bn.moving_variance',   # *****
    'bn1.running_mean': 'bn1.moving_mean',
    'bn1.running_var': 'bn1.moving_variance',
    'bn2.running_mean': 'bn2.moving_mean',
    'bn2.running_var': 'bn2.moving_variance',
    'bn3.running_mean': 'bn3.moving_mean',
    'bn3.running_var': 'bn3.moving_variance',
    'downsample.1.running_mean': 'downsample.1.moving_mean',
    'downsample.1.running_var': 'downsample.1.moving_variance',
    'downsample.0.weight': 'downsample.1.weight',
    'downsample.1.bias': 'downsample.1.beta',
    'downsample.1.weight': 'downsample.1.gamma',
    #'0.bias': '0.beta'
}

match = [
    # feature_map.backbone.
    'focus', 
    'conv1', 
    'csp2',
    'conv3',
    'csp4',
    'conv5',
    'csp6',
    'conv7',
    'spp8',
    'csp9',
    # feature_map.
    'conv10',
    '***',
    '***',
    'csp13',
    'conv14',
    '***',
    '***',
    'csp17',
    'conv18',
    '***',
    'csp20',
    'conv21',
    '***',
    'csp23',
    'back_block1.conv24',  # ××××
    'conv25',
    '***',
    'csp27',
    'conv28',
    '***',
    'conv30',
    'csp31',
    '***',
    'conv33',
    'conv34',
    '***',
    'csp36',
    'conv37',
    '***',
    'conv39',
    'csp40',
    '***',
    'conv42'    
    ]


def pytorch2mindspore():
    """
    Returns:
        object:
    """
    par_dict = torch.load('./End-to-end.pth', map_location='cpu')['state_dict']
    print(par_dict.keys())
    
    new_params_list = []
    for name in par_dict:
        
        print(name)
        param_dict = {}
        parameter = par_dict[name]
        #print(name)
        for fix in param:
            if name.endswith(fix):
                name = name[:name.rfind(fix)]
                name = name + param[fix]

        #print('========================ibn_name', name)

        param_dict['name'] = name
        
        param_dict['data'] = Tensor(parameter.numpy())
        #print(param_dict['data'])
        new_params_list.append(param_dict)

    save_checkpoint(new_params_list, 'weights/End-to-end.ckpt')
    

def ckpt_change():
    checkpoint = ms.load_checkpoint('weights/End-to-end.ckpt')
    
    new_names = []
    old_names = []
    ln = []
    new_params_list = []

    ns = [1, 1, 2, 2, 3, 3]
    ii = 0
    for name in checkpoint.keys():
        old_names.append(name)
        param_dict = {}
        try:
            ln = int(name[6:8])
        except:
            ln = int(name[6:7])
            
    #     print(ln)
            
        str1 = 'model.' + str(ln)
        if ln == 24:
            if '.m.' in name:
                str1 =  f'model.24.m.{ns[ii]-1}'
                str2 = 'feature_map.back_block' + str(ns[ii]) + '.conv24'
        #             print(name)
        #             print(str1)
        #             print(str2)         
                
                ii = ii + 1
        elif ln <= 9:
            str2 = 'feature_map.backbone.' + match[ln]
        else:
            str2 = 'feature_map.' + match[ln]
            
        if 'anchor' in name:
                str1 =  'model.24'
                str2 = 'feature_map.back_block'
        
        new = name.replace(str1, str2)
        new_names.append(name.replace(str1, str2))
        
        
        
        parameter = checkpoint[name].asnumpy()
        
        param_dict['name'] = new
        param_dict['data'] = ms.Tensor(parameter)
        
        #print(param_dict['data'])
        new_params_list.append(param_dict)
    
    
    save_checkpoint(new_params_list, 'weights/my_End-to-end.ckpt')


pytorch2mindspore()
ckpt_change()







#from mindspore.train.serialization import save_checkpoint
#from mindspore import Tensor
#import torch
#def pytorch2mindspore('res18_py.pth'):

#    par_dict = torch.load('res18_py.pth')['state_dict']

#    new_params_list = []

#    for name in par_dict:
#        param_dict = {}
#        parameter = par_dict[name]

#        print('========================py_name',name)
#        if name.endswith('normalize.bias'):
#            name = name[:name.rfind('normalize.bias')]
#            name = name + 'normalize.beta'
#        elif name.endswith('normalize.weight'):
#            name = name[:name.rfind('normalize.weight')]
#            name = name + 'normalize.gamma'
#        elif name.endswith('.running_mean'):
#            name = name[:name.rfind('.running_mean')]
#            name = name + '.moving_mean'
#        elif name.endswith('.running_var'):
#            name = name[:name.rfind('.running_var')]
#            name = name + '.moving_variance'
#        print('========================ms_name',name)

#        param_dict['name'] = name
#        param_dict['data'] = Tensor(parameter.numpy())
#        new_params_list.append(param_dict)

#    save_checkpoint(new_params_list,  'res18_ms.ckpt')
