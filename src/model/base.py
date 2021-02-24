import logging
import torch
import torchvision.models

LOG = logging.getLogger(__name__)


#--------------------------------------------
# helpers

def use_pifpaf_bn(module):
    if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
        module.momentum = 0.01
        module.eps = 1e-4
        module.eval()

def freeze_bn(module):
    if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
        module.eval()
        
class ResnetBlocks():
    def __init__(self, resnet):
        self.modules = list(resnet.children())
        #LOG.debug('modules = %s', self.modules)

    def input_block(self, use_pool=False, conv_stride=2, pool_stride=2):
        modules = self.modules[:4]

        if not use_pool:
            modules.pop(3)
        else:
            if pool_stride != 2:
                modules[3].stride = torch.nn.modules.utils._pair(pool_stride)  # pylint: disable=protected-access

        if conv_stride != 2:
            modules[0].stride = torch.nn.modules.utils._pair(conv_stride)  # pylint: disable=protected-access

        return torch.nn.Sequential(*modules)

    def block2(self):
        return self.modules[4]

    def block3(self):
        return self.modules[5]

    def block4(self):
        return self.modules[6]

    def block5(self):
        return self.modules[7]
    
    def build_backbone(self, use_pool=False,use_last_block = True, pifpaf_bn = True):
        input_modules = self.modules[:4]
        if not use_pool:
            input_modules.pop(3)
            
        input_block = torch.nn.Sequential(*input_modules)
        block2 = self.modules[4]
        block3 = self.modules[5]
        block4 = self.modules[6]
        block5 = self.modules[7]
        
        if use_last_block:
            backbone_net = torch.nn.Sequential(
                           input_block,
                           block2,
                           block3,
                           block4,
                           block5)
        else:
            backbone_net = torch.nn.Sequential(
                           input_block,
                           block2,
                           block3,
                           block4)
        if pifpaf_bn:
            backbone_net.apply(use_pifpaf_bn)

        
        
        # return backbone
        return backbone_net