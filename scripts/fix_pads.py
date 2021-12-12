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

for i in range(0,len(model.graph.node)):
    #print(model.graph.node[i])
    node=model.graph.node[i]
    for j in node.attribute:
        #print(j)
        #print(j.name)
        if(j.name=="pads" and len(j.ints)==4):
            ints=j.ints
            if(ints[0]!=ints[2] or ints[1]!=ints[3]):
                print("found asym pads,trying to fix")
                print(ints)
                ints[0]=max(ints[0],ints[2])
                ints[1]=max(ints[1],ints[3])
    #print("------------")
    #print(numpy_helper.to_array(model.graph.initializer[i]))
    #model.graph.initializer[i].CopyFrom(tensor)
    #print(numpy_helper.to_array(model.graph.initializer[i]))
    #print(model.graph.initializer[i])

onnx.checker.check_model(model)



onnx.save(model, sys.argv[2])
