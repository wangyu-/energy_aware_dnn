import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import torch.onnx as torch_onnx
import numpy

torch.manual_seed(0)

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(3,3), stride=10, padding=0,bias=False)
        self.conv2 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(3,3), stride=10, padding=0)
        self.bn=nn.BatchNorm2d(64)
        self.fc=nn.Linear(30976, 30)

    def forward(self, inputs):
        x = self.conv(inputs)
        y= self.conv2(inputs)
        (x1,x2)=torch.split(x,[32,32],dim=1);
        y=x2.add(y)
        y=torch.cat((x1,y),dim=1)

        x = F.relu(y)
        #y= F.relu(y)
        #z=x.add(y)
        a=torch.flatten(x,start_dim=1)
        #a=z
        a=self.fc(a)
        return a

# Use this an input trace to serialize the model
input_shape = (3, 222, 222)
model_onnx_path = "torch_model.onnx"
model = Model()
model.train(False)

numpy.random.seed(0)
input1=numpy.random.rand(1,3,222,222).astype(numpy.float32)
input1=numpy.full((1,3,222,222),0.1234).astype(numpy.float32)
#input1=numpy.ones((1,3,222,222)).astype(numpy.float32)
input2=torch.from_numpy(input1)
print(model.training)


# Export the model to an ONNX file
dummy_input = Variable(torch.randn(1, *input_shape))
#dummy_input=input1
print("Export of torch_model.onnx complete!")
#print(output)
print(model(input2))

model.train()
print(model(input2))
print(model(input2))
output = torch_onnx.export(model, 
                          dummy_input, 
                          model_onnx_path, 
                          #verbose=True,do_constant_folding=False,opset_version=12)
                          verbose=True,do_constant_folding=False,training=torch.onnx.TrainingMode.TRAINING,opset_version=9)
model.eval()
print(model(input2))
#device = torch.device("cuda:0")
#model.to(device)
#print(model(input2.to(device)))

