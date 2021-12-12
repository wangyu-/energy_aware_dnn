
#module add cuda/10.0.130-gcc/7.1.0
module add cuda/11.0.3-gcc/8.3.1
export LD_LIBRARY_PATH=/home/yw7/TensorRT-8.0.1.6/lib/:/home/yw7/cuda_cudnn-11.3-v8.2.1/lib64:$LD_LIBRARY_PATH
~/TensorRT-8.0.1.6/bin/trtexec --onnx=$1 --saveEngine=1.trt --explicitBatch --workspace=8000  --avgRuns=1000 --duration=30 $2 $3
