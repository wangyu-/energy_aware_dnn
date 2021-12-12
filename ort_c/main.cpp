#include <assert.h>
#include <vector>
#include <onnxruntime_cxx_api.h>
#include <time.h>
#include <cuda_provider_factory.h>

inline long long get_current_time_ms()
{
        timespec tmp_time;
        clock_gettime(CLOCK_MONOTONIC, &tmp_time);
        return tmp_time.tv_sec*1000ll+tmp_time.tv_nsec/(1000*1000l);
}


int main(int argc, char* argv[]) {
  Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "test");
  Ort::SessionOptions session_options;
	OrtSessionOptionsAppendExecutionProvider_CUDA(session_options,0);
//OrtSessionOptionsAppendExecutionProvider_CUDA(session_options, 0)
  //session_options.SetIntraOpNumThreads(1);
  
  //session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

  const char* model_path = "/home/yw7/new_onnx/noopt_sq_fixpad.onnx";

  Ort::Session session(env, model_path, session_options);
  // print model input layer (node names, types, shape etc.)
  Ort::AllocatorWithDefaultOptions allocator;

  // print number of model input nodes
  size_t num_input_nodes = session.GetInputCount();
  std::vector<const char*> input_node_names = {"Placeholder:0"};
  std::vector<const char*> output_node_names = {"avgpool_150:0"};
    
	int batch=1;
  std::vector<int64_t> input_node_dims = {batch,3,222, 222};
  size_t input_tensor_size = batch*3*222*222; 
  std::vector<float> input_tensor_values(input_tensor_size);
  for (unsigned int i = 0; i < input_tensor_size; i++)
    input_tensor_values[i] = 1.0f;
  // create input tensor object from data values
  auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info, input_tensor_values.data(), input_tensor_size, input_node_dims.data(), 4);
  assert(input_tensor.IsTensor());

    
  std::vector<Ort::Value> ort_inputs;
  ort_inputs.push_back(std::move(input_tensor));
  // score model & input tensor, get back output tensor
  for(int i=0;i<10;i++)
	{
		auto a=get_current_time_ms();
		for(int j=0;j<1000;j++)
		{
			//auto result=
  			session.Run(Ort::RunOptions{nullptr}, input_node_names.data(), ort_inputs.data(), ort_inputs.size(), output_node_names.data(), 1);
		}
		auto b=get_current_time_ms();
		printf("%lld\n",b-a);
	}
  auto output_tensors = session.Run(Ort::RunOptions{nullptr}, input_node_names.data(), ort_inputs.data(), ort_inputs.size(), output_node_names.data(), 1);
  
  // Get pointer to output tensor float values
  float* floatarr = output_tensors[0].GetTensorMutableData<float>();
	for(int i=0;i<batch*1000;i++)
	{
		printf("<%f>",floatarr[i]);
	}
  
  printf("Done!\n");
  return 0;
}

