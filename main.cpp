#include "dnn_api.h"

using namespace std;

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

	//cout<<CUDNN_MAJOR<<"."<<CUDNN_MINOR<<endl;
	//int cuda_v;
	//cudaRuntimeGetVersion(&cuda_v);
	//cout<<cuda_v<<endl;

	init();

	Graph graph;
	double time1=get_current_time_ms();
	graph.from_onnx(argv[1]);
	double time2=get_current_time_ms();
	cout<<"from onnx: "<<time2-time1<<endl;
	cout<<"node: "<<graph.nodes.size()<<endl;
	cout<<"tensor: "<<graph.tensors.size()<<endl;
	cout<<"node+tensor: "<<graph.nodes.size()+graph.tensors.size()<<endl;

	Visualizer visualizer;
	visualizer.set_compact_mode(true);
	visualizer.set_show_measure(false);
	visualizer.visualize(graph,"0.html");

	Optimizer optimizer;
	//optimizer.set_trans("rule");
	//optimizer.set_mem("none");
	//optimizer.set_select("none");
	//optimizer.set_select("global");
	//optimizer.set_profiler("Simple");
	double n1=0.58,n2=110.0;
	double w1=0.0,w2=1.0;
	auto cost_func=[&](const measure_t& m){return w1*m.runtime/n1 +w2*m.energy/n2;};
	optimizer.cost_func=cost_func;

	double time_o1=get_current_time_ms();
	graph=optimizer.optimize(graph);
	double time_o2=get_current_time_ms();
	cout<<"optimzie: "<<time_o2-time_o1<<endl;

	optimizer.profiler->fill_measure_info_full(graph);

	visualizer.visualize(graph,"1.html");

	double time_a1=get_current_time_ms();
	graph.save_model("1.model");
	double time_a2=get_current_time_ms();
	cout<<"save model: "<<time_a2-time_a1<<endl;
	Graph graph2;
	double time_b1=get_current_time_ms();
	graph2.load_model("1.model");
	double time_b2=get_current_time_ms();
	cout<<"load model: "<<time_b2-time_b1<<endl;
	double time_c1=get_current_time_us();
	visualizer.visualize(graph2,"graph2.html");
	double time_c2=get_current_time_us();
	cout<<"visualizer: "<<time_c2-time_c1<<endl;
	graph=graph2;

	extern set<string> measure_set;

	cout<<"measure count:"<<measure_set.size()<<endl;
	cout<<"est measure time:"<<(measure_set.size()*(idle_time+stress_time+measure_time)/1000.0+(time_o2-time_o1)/1000)/60<<endl;
	//cout<<"measure count:"<<DB::get_instance()->used.size()<<endl;

	Engine engine;
	double time_d1=get_current_time_us();
	engine.from_graph(graph);
	double time_d2=get_current_time_us();
	cout<<"engine: "<<time_d2-time_d1<<endl;

	auto & in_tensor=engine.get_input(0);
	auto & out_tensor=engine.get_output(0);

	vector<DATATYPE> input=vec_from_file("input.dump");
	assert((int)input.size()==in_tensor.shape.size());
	vector<DATATYPE> output(out_tensor.shape.size());

	in_tensor.host_to_device_async(input.data());
	engine.inference();
	out_tensor.device_to_host_async(output.data());

	vec_to_file(output,"output1.dump");

	auto r2=optimizer.profiler->model(graph);
	cout<<"cost model:"<<r2.to_string()<<endl;

	map<string,int> mp;
	for(auto &x:graph.nodes)
	{
		mp[x.second.extra.algo_name];
	}
	for (auto x:mp)
	{
		printf("<<<%s>>>",x.first.c_str());
	}
	printf("\n");


	void_func_t func_data_copy=[&](){
		in_tensor.host_to_device_async(input.data());
		out_tensor.device_to_host_async(output.data());
	};
	{
		auto r=ProfilerManager::get_instance()->get_profiler("Advanced")->do_measure(func_data_copy);
		cout<<"funca:"<<r.to_string()<<endl;
	}
	void_func_t func=[&](){
		in_tensor.host_to_device_async(input.data());
		engine.inference();
		out_tensor.device_to_host_async(output.data());
		checkCUDA(cudaDeviceSynchronize());
	};

	for(int i=0;i<20;i++)
	{
		auto r=ProfilerManager::get_instance()->get_profiler("Advanced")->do_measure(func);
		cout<<"full runtime:"<<r.to_string()<<endl;
	}

	return 0;
}
