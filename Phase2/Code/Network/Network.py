"""
RBE/CS549 Spring 2022: Computer Vision
Homework 0: Alohomora: Phase 2 Starter Code


Colab file can be found at:
    https://colab.research.google.com/drive/1FUByhYCYAfpl8J9VxMQ1DcfITpY8qgsF

Author(s): 
Prof. Nitin J. Sanket (nsanket@wpi.edu), Lening Li (lli4@wpi.edu), Gejji, Vaishnavi Vivek (vgejji@wpi.edu)
Robotics Engineering Department,
Worcester Polytechnic Institute


Code adapted from CMSC733 at the University of Maryland, College Park.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))

def loss_fn(out, labels):
    ###############################################
    # Fill your loss function of choice here!
    LossFn = nn.CrossEntropyLoss()
    loss = LossFn(out, labels)
    return loss

class ImageClassificationBase(nn.Module):
    def training_step(self, batch):
        images, labels = batch 
        out = self(images)                  # Generate predictions
        loss = loss_fn(out, labels) # Calculate loss
        return loss
    
    def validation_step(self, batch):
        images, labels = batch 
        out = self(images)                    # Generate predictions
        loss = loss_fn(out, labels)   # Calculate loss
        acc = accuracy(out, labels)           # Calculate accuracy
        return {'loss': loss.detach(), 'acc': acc}
        
    def validation_epoch_end(self, outputs):
        batch_losses = [x['loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()   # Combine losses
        batch_accs = [x['acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()      # Combine accuracies
        return {'loss': epoch_loss.item(), 'acc': epoch_acc.item()}
    
    def epoch_end(self, epoch, result):
        print("Epoch [{}], loss: {:.4f}, acc: {:.4f}".format(epoch, result['loss'], result['acc']))

##################################
#Basic
##################################  

class CIFAR10Model(ImageClassificationBase):
  def __init__(self, InputSize, OutputSize):
    
    super().__init__()
    

    """
    Inputs: 
    InputSize - Size of the Input
    OutputSize - Size of the Output
    """
      #############################
      # Fill your network initialization of choice here!
         # Fill your network initialization of choice here!
    self.layer1 = nn.Sequential(
        nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU())
    self.layer2 = nn.Sequential(
        nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU(), 
        nn.MaxPool2d(kernel_size = 2, stride = 2))
    self.layer3 = nn.Sequential(
        nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(128),
        nn.ReLU(),nn.MaxPool2d(kernel_size = 10, stride = 10))   
    self.fc = nn.Sequential(
        nn.Linear(128, 512),
        nn.ReLU())
    self.fc1= nn.Sequential(
        nn.Linear(512, OutputSize))

    
          #############################
    
    
    

      
  def forward(self, xb):
        """
        Input:
        xb is a MiniBatch of the current image
        Outputs:
        out - output of the network
        """
        #############################
        # Fill your network structure of choice here!
        #############################
        out = self.layer1(xb)
        out = self.layer2(out)
        out = self.layer3(out)
        # out = self.layer4(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        out = self.fc1(out)
        out = F.softmax(out)
        return out


##################################
#ResNet 18
##################################    

class ResNet(ImageClassificationBase):
    def __init__(self,OutputSize):
        super().__init__()
        
        self.conv_1 = nn.Sequential(nn.Conv2d(3, 64, kernel_size = 3, padding = 1),
                                nn.BatchNorm2d(64),
                                nn.ReLU(),
                                nn.Conv2d(64, 128, kernel_size = 3, padding = 1),
                                nn.BatchNorm2d(128),
                                nn.ReLU(),
                                nn.MaxPool2d(2))

        self.res_1 = nn.Sequential(nn.Conv2d(128, 128, kernel_size = 3, padding = 1),
                                nn.BatchNorm2d(128),
                                nn.ReLU(),
                                nn.Conv2d(128, 128, kernel_size = 3, padding = 1),
                                nn.BatchNorm2d(128),
                                nn.ReLU())

        self.conv_2 = nn.Sequential(nn.Conv2d(128, 256, kernel_size = 3, padding = 1),
                                nn.BatchNorm2d(256),
                                nn.ReLU(),
                                nn.MaxPool2d(2),
                                nn.Conv2d(256, 512, kernel_size = 3, padding = 1),
                                nn.BatchNorm2d(512),
                                nn.ReLU(),
                                nn.MaxPool2d(2))

        self.res_2 = nn.Sequential(nn.Conv2d(512, 512, kernel_size = 3, padding = 1),
                                nn.BatchNorm2d(512),
                                nn.ReLU(),
                                nn.Conv2d(512, 512, kernel_size = 3, padding = 1),
                                nn.BatchNorm2d(512),
                                nn.ReLU())
        
        self.fc = nn.Sequential(  
                                        nn.Linear(512, OutputSize))
        
    def forward(self, xb):
        out = self.conv_1(xb)
        # print(out.shape)
        out = self.res_1(out) + out
        # print(out.shape)
        out = self.conv_2(out)
        # print(out.shape)
        out = self.res_2(out) + out
        # print(out.shape)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out


##################################
#ResNet 34
##################################        
# class ResNet(ImageClassificationBase):
    # def __init__(self, OutputSize):
    #     super().__init__()
    #     #34 layer res_net
    #     self.conv_1 = nn.Sequential(nn.Conv2d(3, 64, kernel_size = 3, padding = 1),
    #                             nn.BatchNorm2d(64),
    #                             nn.ReLU(),
    #                             nn.Conv2d(64, 64, kernel_size = 3, padding = 1),
    #                             nn.BatchNorm2d(64),
    #                             nn.ReLU(),
    #                             nn.Conv2d(64, 128, kernel_size = 3, padding = 1),
    #                             nn.BatchNorm2d(128),
    #                             nn.ReLU(),
    #                             nn.MaxPool2d(2))

    #     self.res_1 = nn.Sequential(nn.Conv2d(128, 128, kernel_size = 3, padding = 1),
    #                             nn.BatchNorm2d(128),
    #                             nn.ReLU(),
    #                             nn.Conv2d(128, 128, kernel_size = 3, padding = 1),
    #                             nn.BatchNorm2d(128),
    #                             nn.ReLU(),
    #                             nn.Conv2d(128, 128, kernel_size = 3, padding = 1),
    #                             nn.BatchNorm2d(128),
    #                             nn.ReLU(),
    #                             nn.Conv2d(128, 128, kernel_size = 3, padding = 1),
    #                             nn.BatchNorm2d(128),
    #                             nn.ReLU())

    #     self.conv_2 = nn.Sequential(nn.Conv2d(128, 256, kernel_size = 3, padding = 1),
    #                             nn.BatchNorm2d(256),
    #                             nn.ReLU(),
    #                             nn.Conv2d(256, 256, kernel_size = 3, padding = 1),
    #                             nn.BatchNorm2d(256),
    #                             nn.ReLU(),
    #                             nn.Conv2d(256, 256, kernel_size = 3, padding = 1),
    #                             nn.BatchNorm2d(256),
    #                             nn.ReLU(),
    #                             # nn.MaxPool2d(2),
    #                             nn.Conv2d(256, 256, kernel_size = 3, padding = 1),
    #                             nn.BatchNorm2d(256),
    #                             nn.ReLU(),
    #                             nn.Conv2d(256, 256, kernel_size = 3, padding = 1),
    #                             nn.BatchNorm2d(256),
    #                             nn.ReLU(),
    #                             nn.Conv2d(256, 512, kernel_size = 3, padding = 1),
    #                             nn.BatchNorm2d(512),
    #                             nn.ReLU())

    #     self.res_2 = nn.Sequential(nn.Conv2d(512, 512, kernel_size = 3, padding = 1),
    #                             nn.BatchNorm2d(512),
    #                             nn.ReLU(),
    #                             nn.Conv2d(512, 512, kernel_size = 3, padding = 1),
    #                             nn.BatchNorm2d(512),
    #                             nn.ReLU(),
    #                             nn.Conv2d(512, 512, kernel_size = 3, padding = 1),
    #                             nn.BatchNorm2d(512),
    #                             nn.ReLU())
    #     self.avgpool = nn.AvgPool2d(5)
    #     self.fc= nn.Sequential(nn.Linear(512, OutputSize)) 
        
    # def forward(self, xb):
    #     out = self.conv_1(xb)
    #     # print(out.shape)
    #     out = self.res_1(out) + out
    #     # print(out.shape)
    #     out = self.conv_2(out)
    #     # print(out.shape)
    #     out = self.res_2(out) + out
    #     out = self.avgpool(out)
    #     # print(out.shape)
    #     out = torch.flatten(out, 1)
    #     # print(out.shape)
    #     out = self.fc(out)
    #     # print(out.shape)
    #     return out

##################################
#DenseNet
################################## 
class block(nn.Module):
    def __init__(self, in_channels,growth_rate,bottleneck_width):
        super().__init__()
        # print("Inchannels in block :",in_channels)
        self.layers = nn.Sequential(
                                    nn.BatchNorm2d(in_channels),
                                    nn.ReLU(),
                                    nn.Conv2d(in_channels, bottleneck_width*growth_rate, kernel_size=1, padding=0),          #1x1 convolution
                                    nn.BatchNorm2d(bottleneck_width*growth_rate),
                                    nn.ReLU(),
                                    nn.Conv2d(bottleneck_width*growth_rate, growth_rate, kernel_size=3, padding=1))                    #3x3 convolution
              

    def forward(self, xb):
        
        out = self.layers(xb)        
        out = torch.cat([xb, out], 1)            
        return out


class DenseNet(ImageClassificationBase):

    def __init__(self):
        super().__init__()
        growth_rate = 6                                                                
        num_layers= 6                                                           
        num_features= 12
        bottleneck_width = 4
        self.conv1 = nn.Conv2d(3, 12, kernel_size = 1)  
                      
        self.dense1 = self.dense(num_features,growth_rate, num_layers,bottleneck_width)
        num_features += num_layers * growth_rate  
        # print(num_features)
        self.trans1 = nn.Sequential(nn.BatchNorm2d(num_features),                    
                                        nn.Conv2d(num_features, 48, kernel_size=1, padding=0),
                                        nn.AvgPool2d(2))
        num_features= 48
        self.dense2 = self.dense(num_features, growth_rate, num_layers,bottleneck_width)
        num_features += num_layers * growth_rate 
        print(num_features)
        self.trans2 = nn.Sequential(nn.BatchNorm2d(num_features),
                                        nn.ReLU(),
                                        nn.Conv2d(num_features, 96, kernel_size=1, padding=0)
                                        )

        num_features= 96
        self.dense3 = self.dense(num_features,growth_rate, num_layers,bottleneck_width)
        num_features += num_layers * growth_rate
        print(num_features)
        self.classifier = nn.Sequential(nn.BatchNorm2d(num_features),            
                                        nn.AvgPool2d(5),
                                        nn.Flatten(),
                                        nn.Linear(132, 10))

    def dense(self, features, grate, num_layers,bottleneck_width):                            
        layers = []
        for i in range(num_layers):
            layers.append(block(features, grate,bottleneck_width))
            features += grate
            # print("features :",features)
        # print(layers)
        return nn.Sequential(*layers)

    def forward(self, xb):
        out = self.conv1(xb)
        # print(out.shape)
        out = self.dense1(out)
        # print(out.shape)
        out = self.trans1(out)
        # print(out.shape)
        out = self.dense2(out)
        # print(out.shape)
        out = self.trans2(out)
        # print(out.shape)
        out = self.dense3(out)
        # out = torch.flatten(out, 1)
        # print(out.shape)
        out = self.classifier(out)
        # print(out.shape)

        return out


##################################
#ResNeXT
##################################

class res_block(nn.Module):
    def __init__(self, in_features, num_layers,inplanes, stride):
        super().__init__()
        features = num_layers*inplanes             
        self.layers = nn.Sequential(nn.Conv2d(in_features, features, kernel_size = 1),
                                    nn.BatchNorm2d(features),
                                    nn.Conv2d(features, features, kernel_size = 3, stride = stride, padding = 1, groups = num_layers), #2 groups made(as C=2 in our case)
                                    nn.BatchNorm2d(features),
                                    nn.Conv2d(features, 2*features, kernel_size = 1),
                                    nn.BatchNorm2d(2*features))
        self.residual = nn.Sequential(nn.Conv2d(in_features, 2*features, kernel_size = 1, stride = stride),
                                    nn.BatchNorm2d(2*features))
    
    def forward(self, xb):
        out = self.layers(xb)
        out += self.residual(xb)
        out = F.relu(out)
        return out

class ResNeXt(ImageClassificationBase):

    def __init__(self):
        super().__init__()
        self.in_channels = 64
       
        self.num_layers = 2
        self.in_features = 64
        self.bottleneck_width = 2

        self.conv1 = nn.Conv2d(3, 64, kernel_size = 1)
        self.bn1 = nn.BatchNorm2d(64)
        self.block1 = self.block(self.in_channels, self.num_layers, self.in_features, 1)
        self.in_channels = self.bottleneck_width*self.num_layers*self.in_features
        self.in_features = self.in_features*2
        self.block2 = self.block(self.in_channels, self.num_layers, self.in_features,2) 
        self.in_channels = self.bottleneck_width*self.num_layers*self.in_features
        self.in_features = self.in_features*2          
        self.block3 = self.block(self.in_channels, self.num_layers, self.in_features,2)
        
        self.avgpool = nn.AvgPool2d(8)
        # self.batchnorm=nn.BatchNorm2d(1024)
        self.fc = nn.Sequential(nn.Linear(1024,10))
        # self.fc1 =nn.Sequential( nn.Linear(9126,10))                               



    def block(self,in_,nlayers,inf, stride):
        layers = []
        layers.append(res_block(in_, nlayers, inf, stride))        
        return nn.Sequential(*layers)
        
    def forward(self,xb):
        out = self.conv1(xb)
        # print(out.shape)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.block1(out)
        # print(out.shape)
        out = self.block2(out)
        # print(out.shape)
        out = self.block3(out)
        out= self.avgpool(out)
        # out= self.batchnorm(out)
        out = torch.flatten(out, 1) 
        
        # print(out.shape)
        out = self.fc(out)

        return out
