//#include <fstream>
//#include <iostream>
//#include <cassert>

#include "dnn_api.h"

using namespace std;

//vector<float> global_data;
int disable_copy=0;  //for concat concat2 split
int disable_icopy=0; //identity copy
int protect_input_output=0;

int main(int argc, char **argv)
{
  if(argc<=1)
  {
    mylog(log_fatal,"no file argument provided\n");
    exit(0);
  }
  init();

  //cout<<in_graph.DebugString()<<endl;
  //global_data=vec_from_file("input.dump");

  printf("0\n");

  Graph graph;
  graph.from_onnx(argv[1]);

  //graph.post_process();

  //ConvRelu s1;
  //SplitConcat s2;
  //s1.apply_all_inplace(graph);
  ////cost_func_t cost_func=[&](const measure_t& m){return 1*m.runtime/0.58+0.0*m.energy/110;};
  ///string profiler_name="Advanced";
  

  //graph.tensors.at("ew_add_108:0").read_by.clear();
  //graph.outputs["ew_add_108:0"];

  Visualizer visualizer;
  visualizer.set_compact_mode(true);
  visualizer.visualize(graph,"0.html");

  ////graph.memory_copy_optimize();
  //s2.apply_all_inplace(graph);
  //graph.to_html("01.html");
/*
  auto sub=graph.subgraph({"conv2d_116_relu:0","conv2d_118_relu:0"},{"ew_add_119:0"});
  //enlarge_conv(sub,"conv2d_227_bias_add",{3,3});
  //enlarge_conv(sub,"conv2d_228_bias_add",{3,3});
  //merge_conv_neq(sub,"conv2d_227_bias_add","conv2d_228_bias_add");
  //auto sub=graph.subgraph({"14","15"},{"19"});
  //SplitSplit sss;
  //sss.apply_all_inplace(sub);
  ConvAdd sss;
	sss.apply_all_inplace(sub);
  sub.to_html("sub.html");*/
  //auto sub=graph.subgraph({"conv2d_103_relu:0"},{"concat_106:0"});
  
  //auto sub=graph.subgraph({"maxpool_102:0"},{"ew_add_108:0"});
  /*
  auto sub=graph.subgraph({"conv2d_103_relu:0"},{"conv2d_104_relu:0","conv2d_105_relu:0"});
  //s1.apply_all_inplace(sub);
 
  //merge_conv(sub,"conv2d_103_bias_add","conv2d_107_bias_add");
  //enlarge_conv(sub,"conv2d_105_bias_add",{3,3});
  merge_conv_neq(sub,"conv2d_104_bias_add","conv2d_105_bias_add");
  sub.memory_copy_optimize();
  //s2.apply_all_inplace(sub);
  SelectorManager::get_instance()->get_selector("Local")->select_impl(sub,profiler_name,cost_func);
  fill_measure_info(sub,profiler_name);
  sub.to_html("sub.html");*/


  //greedy_fix_point(graph);
  //merge_conv_neq(graph,"conv2d_227_bias_add","conv2d_228_bias_add");
  //greedy_fix_point(graph);
  //graph.memory_copy_optimize();
  
  Optimizer optimizer;
  //optimizer.set_trans("rule");
  //optimizer.set_mem("none");
  //optimizer.set_select("none");
  optimizer.set_select("global");
  auto cost_func=[&](const measure_t& m){return m.runtime;};
  optimizer.cost_func=cost_func;
  graph=optimizer.optimize(graph);

  //optimizer.profiler->fill_measure_info_full(graph);
  
  
  //optimize(graph,profiler_name,cost_func);
  //graph.resolve_lazy();

  visualizer.visualize(graph,"1.html");

  graph.save_model("1.model");

  Graph graph2;
  graph2.load_model("1.model");

  visualizer.visualize(graph2,"graph2.html");

  graph=graph2;

  //graph.resolve_lazy();


  //cudnnWorkspaceManager::get_default();

  Engine engine;
  //SelectorManager::get_instance()->get_selector("Local")->select_impl(graph,profiler_name,cost_func);
  engine.from_graph(graph);

  auto & in_tensor=engine.get_input(0);
  auto & out_tensor=engine.get_output(0);

  //engine.tensors[in].alloc_on_host();
  //for(int i=0;i<engine.tensors[in].data.size();i++)
  vector<DATATYPE> input=vec_from_file("input.dump");
  assert((int)input.size()==in_tensor.shape.size());
  vector<DATATYPE> output(out_tensor.shape.size());

  in_tensor.host_to_device_async(input.data());
  engine.inference();
  out_tensor.device_to_host_async(output.data());

  vec_to_file(output,"output1.dump");


/*
  fill_measure_info(graph,profiler_name);
  fill_measure_info_full(graph,profiler_name);

  visualizer.visualize(graph,"extra.html");
*/

  auto r2=optimizer.profiler->model(graph);
  cout<<"cost model:"<<r2.to_string()<<endl;
  //printf("9\n");

  //disable_copy=1;
  //disable_icopy=1;

 // printf("8\n");
  void_func_t func=[&](){
			in_tensor.host_to_device_async(input.data());
			engine.inference();
			out_tensor.device_to_host_async(output.data());
			//checkCUDA(cudaStreamSynchronize(engine.stream));
			checkCUDA(cudaDeviceSynchronize());
			};


  ProfilerManager::get_instance()->get_profiler("Simple")->do_measure(func);
  for(int i=0;i<20;i++)
  {
		  auto r=ProfilerManager::get_instance()->get_profiler("Advanced")->do_measure(func);
		  cout<<"full runtime:"<<r.to_string()<<endl;
  }

  return 0;
}
