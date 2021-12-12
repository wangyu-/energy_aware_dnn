import onnx
from onnx import numpy_helper
import numpy
import sys

# Load the ONNX model
model = onnx.load(sys.argv[1])

# Check that the IR is well formed
print("check before fix pads")
onnx.checker.check_model(model)

# Print a human readable representation of the graph

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

print("check after fix pads")
onnx.checker.check_model(model)

onnx.save(model, "tmp.onnx")

#print(onnx.helper.printable_graph(model.graph))

output =[node.name for node in model.graph.output]

input_all = [node.name for node in model.graph.input]
input_initializer =  [node.name for node in model.graph.initializer]
net_feed_input = list(set(input_all)  - set(input_initializer))

print('Inputs: ', net_feed_input)
print('Outputs: ', output)

mp={}
for node in model.graph.input:
    #print(node.name)
    if node.name in net_feed_input:
        mp[node.name]=node

input0_name=net_feed_input[0]
onnx_shape=mp[input0_name].type.tensor_type.shape
shape=[]
#print(onnx_shape.dim)
for x in onnx_shape.dim:
    #print(x.dim_value)
    shape.append(x.dim_value)

print(shape)

######################################################################################
import onnxruntime as ort
import sys
EP_list = ['CPUExecutionProvider']
ort_session = ort.InferenceSession("tmp.onnx", providers=EP_list)
print("111")

#ort_session.set_providers(['CPUExecutionProvider'])

def dump_to_file(t,filename):
    t=list(t.flat);
    print("-----")
    f = open(filename, "w")
    f.write(str(len(t)));
    f.write("\n")
    for i in t:
        f.write("%.6g"%(i,))
        f.write("\n")


import numpy
numpy.random.seed(0)
input0=numpy.random.rand(*shape).astype(numpy.float32)
input0=numpy.ones(shape).astype(numpy.float32)

outputs = ort_session.run(None, {input0_name: input0})

#print(outputs[0])
dump_to_file(input0,"input.dump")
dump_to_file(outputs[0] ,"output.dump")
