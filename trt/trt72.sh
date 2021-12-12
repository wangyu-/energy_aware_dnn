
#module add cuda/10.0.130-gcc/7.1.0
export LD_LIBRARY_PATH=/home/yw7/TensorRT-7.2.3.4/lib:/home/yw7/cuda_cudnn-10.2-v8.2.2/lib64:$LD_LIBRARY_PATH
#export LD_LIBRARY_PATH=/home/yw7/TensorRT-7.2.3.4/lib::/home/yw7/cuda_cudnn-11.3-v8.2.1/lib64::$LD_LIBRARY_PATH
~/TensorRT-7.2.3.4/bin/trtexec --onnx=$1 --saveEngine=1.trt --explicitBatch --workspace=8000  --avgRuns=1000 --duration=30 $2 $3
