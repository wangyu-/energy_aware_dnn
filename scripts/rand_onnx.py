import onnx
from onnx import numpy_helper
import numpy
import sys

# Load the ONNX model
model = onnx.load(sys.argv[1])

# Check that the IR is well formed
onnx.checker.check_model(model)

# Print a human readable representation of the graph
print(onnx.helper.printable_graph(model.graph))

for i in range(0,len(model.graph.initializer)):
    #print("------------")
    #print(numpy_helper.to_array(model.graph.initializer[i]))
    shape=numpy_helper.to_array(model.graph.initializer[i]).shape
    name=model.graph.initializer[i].name
    #print(name)
    #print(model.graph.initializer[i])
    arr=numpy.random.normal(loc=0.0,scale=0.3,size=shape).astype(numpy.float32)
    tensor = numpy_helper.from_array(arr,name)
    model.graph.initializer[i].CopyFrom(tensor)
    #print(numpy_helper.to_array(model.graph.initializer[i]))
    #print(model.graph.initializer[i])



onnx.save(model, sys.argv[2])
