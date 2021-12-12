
module add cuda/10.0.130-gcc/7.1.0
export LD_LIBRARY_PATH=/home/yw7/TensorRT-7.0.0.11/lib/:$LD_LIBRARY_PATH
~/TensorRT-7.0.0.11/bin/trtexec --onnx=$1 --saveEngine=1.trt --explicitBatch --workspace=8000  --avgRuns=1000 --duration=30 $2 $3
