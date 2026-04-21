#!/usr/bin python
import torch
from torch import nn
from torch.nn import functional as F
from torchvision import models

import config

class Net_test(nn.Module):
    def __init__(self, **params):
        super(Net_test, self).__init__()
        self.out_dim = params['out_dim']
        
        features_layers = [ nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
                            nn.BatchNorm2d(64),
                            nn.ReLU(inplace=True),
                            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)]
        self.features = nn.Sequential(*features_layers)

        test_params = list(map(id, self.features.parameters()))
        print(test_params)

        
class VGG(nn.Module):
    def __init__(self, init_weights=True, freezing=True, **params):
        # Using the pre-training model provided by pytorch
        super(VGG, self).__init__()

        # parameters setting
        self.cnn_out_dim = params['out_dim']

        # utilize vgg16 pre-trained model to extract features and classifying
        pretrained_cnn = models.vgg16(pretrained=True)
        
        # features
        cnn_feature_layers = list(pretrained_cnn.features)
        self.features = nn.Sequential(*cnn_feature_layers)
        
        # classifier
        cnn_classify_layers = list(pretrained_cnn.classifier)[:-1]
        self.fc1 = nn.Sequential(*cnn_classify_layers)
        
        # new layer in classifier
        in_features = 4096
        self.fc_final = nn.Sequential(
            *[nn.Linear(in_features, self.cnn_out_dim)
                ])

        #cnn_classify_layers += self.fc_final
        #self.classifier = nn.Sequential(*cnn_classify_layers)

        if freezing:
            self._freeze()
            
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        features = self.fc1(x)
        output = self.fc_final(features)
        #x = F.softmax(x, dim=0)
        
        return features, output

    def _initialize_weights(self):
        for child_index, child in enumerate(self.children()):
            if child_index == 0:
                for m_index, m in enumerate(child):
                    if m_index >= 24:
                        for param in m.parameters():
                            param.requires_grad = True
                            
                        if isinstance(m, nn.Conv2d):
                            # kaiming normal
                            nn.init.kaiming_normal_(
                               m.weight, mode='fan_out', nonlinearity='relu')
                            if m.bias is not None:
                                nn.init.constant_(m.bias, 0)
                        elif isinstance(m, nn.BatchNorm2d):
                            nn.init.constant_(m.weight, 1)
                            nn.init.constant_(m.bias, 0)
                        elif isinstance(m, nn.Linear):
                            nn.init.normal_(m.weight, 0, 0.01)
                            nn.init.constant_(m.bias, 0)
            else:
                for m_index, m in enumerate(child):
                    for param in m.parameters():
                        param.requires_grad = True

                    if isinstance(m, nn.Conv2d):
                        # kaiming normal
                        nn.init.kaiming_normal_(
                               m.weight, mode='fan_out', nonlinearity='relu')
                        if m.bias is not None:
                            nn.init.constant_(m.bias, 0)
                    elif isinstance(m, nn.BatchNorm2d):
                        nn.init.constant_(m.weight, 1)
                        nn.init.constant_(m.bias, 0)
                    elif isinstance(m, nn.Linear):
                        nn.init.normal_(m.weight, 0, 0.01)
                        nn.init.constant_(m.bias, 0)

    def _freeze(self):
        for m_index, m in enumerate(self.features):
            if m_index < 24:
                for param in m.parameters():
                    param.requires_grad = False

def main():
    net = VGG(**config.net_params)

    for child_index, child in enumerate(net.children()):
        for m_index, m in enumerate(child):
            for index, parameter in enumerate(m.parameters()):
                print(parameter)
    #print(net.features)
    #print(net.classifier)
    

def test():
    net = Net_test()
    for index, parameter in enumerate(net.parameters()):
        print('index {}'.format(index))
        print(parameter)
    print(net.features)

    
if __name__ == "__main__":
    main()
    #test()
