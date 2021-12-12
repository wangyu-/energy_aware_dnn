import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import torch.onnx as torch_onnx
import numpy

torch.manual_seed(0)

def dump_to_file(t,filename):
    t=list(t.flat);
    print("-----")
    f = open(filename, "w")
    f.write(str(len(t)));
    f.write("\n")
    for i in t:
        f.write(str(i))
        f.write("\n")


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        #self.conv = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(3,3), stride=100, padding=0,bias=False)
        self.conv = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(3,3), stride=1, padding=0)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=2, kernel_size=(3,3), stride=10, padding=0)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=2, kernel_size=(3,3), stride=10, padding=0)
        self.pool= nn.MaxPool2d((3, 3), stride=(2, 2))
        self.pool2= nn.AvgPool2d((3, 3), stride=(2, 2),count_include_pad=False)

    def forward(self, inputs):
        y = self.conv(inputs)
        #x = F.relu(x)
        x1 = self.conv2(y)
        x2 = self.conv3(y)
        x=x1.add(x2)
        x=self.pool(x)
        x=self.pool2(x)
        x = F.relu(x)
        return x

# Use this an input trace to serialize the model
input_shape = (3, 222, 222)
model_onnx_path = "torch_model.onnx"
model = Model()
model.train(False)
BATCH=2
numpy.random.seed(0)
input1=numpy.random.rand(BATCH,3,222,222).astype(numpy.float32)
#input1=numpy.full((BATCH,3,222,222),2.34567).astype(numpy.float32)
#input1=numpy.ones((BATCH,3,222,222)).astype(numpy.float32)
input2=torch.from_numpy(input1)
print(model.training)


# Export the model to an ONNX file
dummy_input = Variable(torch.randn(BATCH, *input_shape))
#dummy_input=input1
print("Export of torch_model.onnx complete!")
#print(output)
#print(model(input2))

model.train()
#print(model(input2))
#print(model(input2))
output = torch_onnx.export(model, 
                          dummy_input, 
                          model_onnx_path, 
                          #verbose=True,do_constant_folding=False,opset_version=12)
                          #verbose=True,do_constant_folding=False,training=torch.onnx.TrainingMode.TRAINING,opset_version=12)
                          verbose=True,do_constant_folding=False,training=torch.onnx.TrainingMode.EVAL)
model.eval()
output2=model(input2)
print(output2)
dump_to_file(input1,"input.dump")
dump_to_file(output2.data.numpy(),"output.dump")
#dump_to_file(output2,"output.dump")


#for name, param in model.state_dict().items():
#    print(name)
