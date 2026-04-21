import torch
from torch import nn
from torch.nn import functional as F
from torchvision import models

import config

class EasyFCNet(nn.Module):
    def __init__(self, init_weights=True, freezing=False, **params):
        # Using the pre-training model provided by pytorch
        super(EasyFCNet, self).__init__()
        
        self.cnn_out_dim_phase = params['out_dim_phase']
        
        # phase_tool layer
        in_features_fc_phase = 4103
        self.fc_phase = nn.Sequential(
            *[nn.Linear(in_features_fc_phase, self.cnn_out_dim_phase)
                ])
        
        if freezing:
            freeze_layers = []
            self._freeze(freeze_layers)
            
        if init_weights:
            initial_layers = [self.fc_phase]
            self._initialize_weights(initial_layers)
    
    def forward(self, X):
        # get the result of EasyFCNet
        phase_output = self.fc_phase(X)
        
        return phase_output
    
    def _initialize_weights(self, initial_layers):
        for idx, layer in enumerate(initial_layers):
            for param in layer.parameters():
                param.requires_grad = True
                        
            if isinstance(layer, nn.Conv2d):
                # kaiming normal
                nn.init.kaiming_normal_(
                        layer.weight, mode='fan_out', nonlinearity='relu')
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)
            elif isinstance(layer, nn.BatchNorm2d):
                nn.init.constant_(layer.weight, 1)
                nn.init.constant_(layer.bias, 0)
            elif isinstance(layer, nn.Linear):
                nn.init.normal_(layer.weight, 0, 0.01)
                nn.init.constant_(layer.bias, 0)
    
class AlexNet(nn.Module):
    def __init__(self, init_weights=True, freezing=False, **params):
        # Using the pre-training model provided by pytorch
        super(AlexNet, self).__init__()

        # parameters setting
        self.cnn_out_dim_tool = params['out_dim_tool']
        
        # utilize alexnet pre-trained model to extract features and classifying
        pretrained_cnn = models.alexnet(pretrained=True)
        
        # features
        cnn_feature_layers = list(pretrained_cnn.features)
        self.features = nn.Sequential(*cnn_feature_layers)
        
        # classifier
        cnn_classifier_layers = list(pretrained_cnn.classifier)[:-1]
        self.fc1 = nn.Sequential(*cnn_classifier_layers)
        
        # fc_tool layer
        in_features_fc_tool = 4096
        self.fc_tool = nn.Sequential(
            *[nn.Linear(in_features_fc_tool, self.cnn_out_dim_tool)
                ])
        
        if freezing:
            freeze_layers = []
            self._freeze(freeze_layers)
            
        if init_weights:
            initial_layers = [self.fc_tool]
            self._initialize_weights(initial_layers)

    def forward(self, X):
        # get the result of AlexNet
        X = self.features(X)
        X = X.view(X.size(0), -1)
        
        # classifier of AlexNet
        X = self.fc1(X)
        
        # get the result of ToolNet
        tool_output_feature = self.fc_tool(X)
        tool_output = torch.sigmoid(tool_output_feature) 
        
        return X, tool_output
    
    def _freeze(self, freeze_layers):
        for m_index, layer in enumerate(freeze_layers):
            for param in layer.parameters():
                param.requires_grad = False
                    
    def _initialize_weights(self, initial_layers):
        for idx, layer in enumerate(initial_layers):
            for param in layer.parameters():
                param.requires_grad = True
                        
            if isinstance(layer, nn.Conv2d):
                # kaiming normal
                nn.init.kaiming_normal_(
                        layer.weight, mode='fan_out', nonlinearity='relu')
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)
            elif isinstance(layer, nn.BatchNorm2d):
                nn.init.constant_(layer.weight, 1)
                nn.init.constant_(layer.bias, 0)
            elif isinstance(layer, nn.Linear):
                nn.init.normal_(layer.weight, 0, 0.01)
                nn.init.constant_(layer.bias, 0)

def main():
    net = AlexNet(**config.net_params)

    for child_index, child in enumerate(net.children()):
        for index, parameter in enumerate(child.parameters()):
            print(parameter)
    
  
if __name__ == "__main__":
    main()
