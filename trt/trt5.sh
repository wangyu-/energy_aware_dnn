module add cuda/10.0.130-gcc/7.1.0
export LD_LIBRARY_PATH=/home/yw7/TensorRT-5.1.5.0/lib/:$LD_LIBRARY_PATH
#~/TensorRT-6.0.1.5//bin/trtexec  --onnx=$1 --saveEngine=1.trt --batch=$2 --workspace=8000 --avgRuns=1000 --duration=30 --verbose
~/TensorRT-5.1.5.0//bin/trtexec  --onnx=$1 --saveEngine=1.trt --batch=$2 --workspace=8000 --avgRuns=1000 --verbose
