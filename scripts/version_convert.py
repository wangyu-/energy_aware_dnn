import onnx
import sys

# Load the model
model = onnx.load(sys.argv[1])

# Check that the IR is well formed
onnx.checker.check_model(model)

model2 = onnx.shape_inference.infer_shapes(model)

from onnx import version_converter

# Convert to version 8
converted_model = version_converter.convert_version(model2, 8)

# Save model
onnx.save(converted_model, sys.argv[2])
