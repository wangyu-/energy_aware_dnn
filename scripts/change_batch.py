import onnx
from onnx import numpy_helper
import numpy
import sys

# Load the ONNX model

model = onnx.load(sys.argv[1])

# Check that the IR is well formed
onnx.checker.check_model(model)

# Print a human readable representation of the graph
#print(onnx.helper.printable_graph(model.graph))

batch_size= int(sys.argv[3])


inputs = model.graph.input
outputs = model.graph.output
inits=set()
for init in model.graph.initializer:
    inits.add(init.name)
for input in inputs:
    if(input.name in inits): 
        continue
    dim1 = input.type.tensor_type.shape.dim[0]
    print(dim1.dim_value,"/",dim1.dim_param)
    dim1.dim_value=batch_size

for output in outputs:
    dim1 = output.type.tensor_type.shape.dim[0]
    print(dim1.dim_value,"/",dim1.dim_param)
    dim1.dim_value=batch_size

onnx.checker.check_model(model)



onnx.save(model, sys.argv[2])
