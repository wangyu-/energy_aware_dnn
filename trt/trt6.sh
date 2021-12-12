module add cuda/10.0.130-gcc/7.1.0
export LD_LIBRARY_PATH=/home/yw7/TensorRT-6.0.1.5/lib/:$LD_LIBRARY_PATH
#~/TensorRT-6.0.1.5//bin/trtexec  --onnx=$1 --saveEngine=1.trt --batch=$2 --workspace=8000 --avgRuns=1000 --duration=30 --verbose
~/TensorRT-6.0.1.5//bin/trtexec  --onnx=$1 --saveEngine=1.trt --batch=$2 --workspace=8000 --avgRuns=3000 --duration=100   $3 $4
