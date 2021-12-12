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

inferred_model = onnx.shape_inference.infer_shapes(model)
print(onnx.helper.printable_graph(inferred_model.graph))

print('After shape inference, the shape info of Y is:\n{}'.format(inferred_model.graph.value_info))

onnx.save(inferred_model, sys.argv[1]+'.shape')
