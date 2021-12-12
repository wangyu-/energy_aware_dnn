#include "engine.h"

ExecuterManager * ExecuterManager::p=0;
cudnnWorkspace * cudnnWorkspace ::p=0;

void DeviceTensor::from_tensor(Tensor &tensor)
{
	bool use_random=false;
	name = tensor.name;
	shape = tensor.shape;
	force_stride_scale = tensor.force_stride_scale;
	auto &inits = tensor.graph->workspace->inits;
	if (inits.find(tensor.name) != inits.end())
	{
		if (force_stride_scale != 1) {mylog(log_fatal, "tensor from initilizer doesn't support stride\n");exit(-1);}

		alloc_on_device();
		
		if (use_random)
		{
			cudaMemcpy(data_ptr, random_pool.data(), shape.byte_size(), cudaMemcpyHostToDevice);
		}
		else
		{
			if(inits[tensor.name].lazy.has_value)
			{
				//mylog(log_fatal,"Inits %s has lazy value, should be resolved first\n",tensor.name.c_str());
				//assert(0==1);
				//exit(-1);
				cudaMemcpy(data_ptr, random_pool.data(), shape.byte_size(), cudaMemcpyHostToDevice); //todo?
			}
			else
			{
					data = inits[tensor.name].data;
					//printf("%s %d %d %d\n",tensor.name.c_str(),(int)data.size(),shape.byte_size(),(int)shape.dims.size());
					assert((int)data.size() == shape.size());
					host_to_device();
			}
		}
	}
	else if ( tensor.mem.part_of == "")	//just a normal intermediate tensor
	{
		alloc_on_device();
		if (use_random)
		{
			cudaMemcpy(data_ptr, random_pool.data(), shape.byte_size(), cudaMemcpyHostToDevice);
		}

	}
	else
	{
		//alloc_on_device();
		//cout<<"special:"<<tensor.name<<endl;

		auto &parent = tensor.graph->tensors.at(tensor.mem.part_of);
		assert(parent.shape.dims.size() == 4);
		assert(tensor.shape.dims.size() == 4);
		auto &device_parent = engine->tensors.at(tensor.mem.part_of);
		if (device_parent.data_ptr == 0)
			return;
		auto strides = parent.shape.get_strides();
		if (device_parent.strides_override.size() != 0)
			strides = device_parent.strides_override;

		int offset = tensor.mem.offset;
		int hw = strides[1];
		strides_override = strides;
		//for(int i=0;i<(int)strides.size();i++)
		//	cout<<strides[i]<<" ";
		//cout<<endl;
		data_ptr = (char *)device_parent.data_ptr + hw * sizeof(DATATYPE) * offset;
	}
}

void Engine::from_graph(Graph &graph)
{
	for(auto &x:graph.nodes)
	{
		assert(x.second.extra.algo_name!="");
	}
	for(auto &x:graph.tensors)
	{
		auto name=x.first;
		auto &device_tensor=tensors[name];
		device_tensor.engine=this;
	}
	for(int round=0;;round++)
	{
		if(round==200) mylog(log_warn,"possible dead loop\n");
		int cnt=0;
		for(auto &x:graph.tensors)
		{
			auto name=x.first;
			auto &device_tensor=tensors.at(name);
			auto &tensor=graph.tensors.at(name);
			if(device_tensor.data_ptr==0)
			{
				device_tensor.from_tensor(x.second);
				cnt++;
			}
		}
		if(cnt==0) break;
	}

	for(auto &x:graph.topo_order)
	{
		string &node_name=x;
		auto & node=graph.nodes[node_name];
		string &node_type=node.type;

		string impl_name=graph.nodes.at(node_name).extra.algo_name;
		Executer* executer=ExecuterManager::get_instance()->get_exec(node.type,impl_name)->clone();
		executers.push_back(executer);
		executers_mp[node_name]=executer;
		executer->engine=this;
		executer->name=node.name;
		executer->init(node);
	}
	inputs.clear();
	outputs.clear();
	for(auto &x:graph.inputs)
	{
		inputs.push_back(x.first);
	}
	for(auto &x:graph.outputs)
	{
		outputs.push_back(x.first);
	}
}

void Engine::from_node(Node &node,const string &impl_name)
{
	Graph new_graph;
	new_graph.from_node(node);

	new_graph.nodes.at(node.name).extra.algo_name=impl_name;
	//map<string,string> impl_mp;
	//impl_mp[node.name]=impl_name;
	from_graph(new_graph);
	for(auto &x:new_graph.tensors)
	{
		if(x.second.mem.part_of=="")   //this include the nodes with initilizers 
		{	
			tensors.at(x.first).fill_with_random();
		}
	}

}

struct Descriptor:NoCopy
{
    //Descriptor()=default;
    //Descriptor(const Descriptor&)=delete;
    //void operator=(const Descriptor&)=delete;
};

struct TensorDescriptor:Descriptor
{
    TensorDescriptor(){cudnnCreateTensorDescriptor(&descriptor);}
    ~TensorDescriptor(){cudnnDestroyTensorDescriptor(descriptor);}
    cudnnTensorDescriptor_t descriptor;
    void *data_ptr=0;
	void set_as_4d_tensor(DeviceTensor &device_tensor)
	{
		assert(device_tensor.shape.dims.size()==4);
		set_as_nd_tensor(device_tensor);
	}
	void set_as_4d_tensor_from_1d(DeviceTensor &device_tensor)
	{
		auto &shape=device_tensor.shape;
		assert(shape.dims.size()==1);
		vector<int> dims={1,shape.dims[0],1,1};
		set_as_nd_tensor(device_tensor,dims);
	}
	void set_as_2d_tensor_from_4d(DeviceTensor &device_tensor)
	{
		auto &shape=device_tensor.shape;
		assert(shape.dims.size()==4);
		vector<int> dims={shape.dims[0],shape.dims[1]*shape.dims[2]*shape.dims[3]};
		set_as_nd_tensor(device_tensor,dims);
	}
	void set_as_4d_tensor_sub(DeviceTensor &device_tensor,int offset,int size)
    {
        assert(data_ptr==0);
		auto &shape=device_tensor.shape;
        assert(shape.dims.size()==4);

		vector<int> old_strides;
		if(device_tensor.strides_override.size()!=0)
		{
			old_strides=device_tensor.strides_override;
		}
		else
		{
			old_strides=Shape::to_strides(shape.dims);
		}

		vector<int> new_dims=shape.dims;
		new_dims[1]=size;

		//vector<int> new_strides=Shape::to_strides(new_dims);
		int hw=old_strides[1];
		//new_strides[0]*=shape.dims[1];
		set_as_nd_tensor(device_tensor,new_dims,old_strides);
		data_ptr=(char*)device_tensor.data_ptr+hw*sizeof(DATATYPE)*offset;

		//checkCUDNN(cudnnSetTensor4dDescriptorEx(descriptor,CUDNN_DATATYPE , n, c, h, w, shape.dims[1]*h*w*stride_scale , h*w,w,1));
		//checkCUDNN(cudnnSetTensor4dDescriptor(descriptor,CUDNN_TENSOR_NCHW,CUDNN_DATATYPE , n, c, h, w));
    }
	void set_as_nd_tensor(DeviceTensor &device_tensor)
	{
		set_as_nd_tensor(device_tensor,device_tensor.shape.dims);
	}
	void set_as_nd_tensor(DeviceTensor &device_tensor,vector<int> dims) //do not use reference
	{
		if(device_tensor.strides_override.size()!=0)
		{
			set_as_nd_tensor(device_tensor,dims,device_tensor.strides_override);
		}
		else
		{
			auto strides=Shape::to_strides(dims);
			//strides[0]*=device_tensor.stride_scale;
			set_as_nd_tensor(device_tensor,dims,strides);
		}
	}
	void set_as_nd_tensor(DeviceTensor &device_tensor,vector<int> dims,vector<int> strides)
	{
		assert(data_ptr==0);
		auto &stride_scale=device_tensor.force_stride_scale;
        	data_ptr=device_tensor.data_ptr;
		assert(dims.size()>=2&&dims.size()<=4);
		assert(dims.size()==strides.size());
		int pad=4-dims.size();
		for(int i=0;i<pad;i++) {dims.push_back(1);strides.push_back(1);}
		strides[0]*=stride_scale;
		checkCUDNN(cudnnSetTensorNdDescriptor(descriptor,CUDNN_DATATYPE ,(int)dims.size(), dims.data(),strides.data()));

	}

};

struct FilterDescriptor:Descriptor
{
    FilterDescriptor(){cudnnCreateFilterDescriptor(&descriptor);}
    ~FilterDescriptor(){cudnnDestroyFilterDescriptor(descriptor);}
    cudnnFilterDescriptor_t descriptor;
    void *data_ptr=0;
};

struct PoolingDescriptor:Descriptor
{
    PoolingDescriptor(){cudnnCreatePoolingDescriptor(&descriptor);}
    ~PoolingDescriptor(){cudnnDestroyPoolingDescriptor(descriptor);}
    cudnnPoolingDescriptor_t descriptor;
};

struct ActivationDescriptor:Descriptor
{
    ActivationDescriptor(){cudnnCreateActivationDescriptor(&descriptor);}
    ~ActivationDescriptor(){cudnnDestroyActivationDescriptor(descriptor);}
    cudnnActivationDescriptor_t descriptor;
};

struct ConvolutionDescriptor:Descriptor
{
    ConvolutionDescriptor(){cudnnCreateConvolutionDescriptor(&descriptor);}
    ~ConvolutionDescriptor(){cudnnDestroyConvolutionDescriptor(descriptor);}
    cudnnConvolutionDescriptor_t descriptor;
};
struct OpTensorDescriptor:Descriptor
{
	OpTensorDescriptor(){cudnnCreateOpTensorDescriptor(&descriptor);}
	~OpTensorDescriptor(){cudnnDestroyOpTensorDescriptor(descriptor);};
	cudnnOpTensorDescriptor_t descriptor;
};
struct ConvImpl:Executer
{
	TensorDescriptor input;
	FilterDescriptor filter;
	TensorDescriptor bias;
	TensorDescriptor add;
	TensorDescriptor output;
	ConvolutionDescriptor conv_desc;
	cudnnConvolutionFwdAlgo_t algo=(cudnnConvolutionFwdAlgo_t)0;
	ActivationDescriptor acti_desc;

	OpTensorDescriptor op_desc;
	cudnnOpTensorOp_t op_type;

	vector<int> kernel_shape;
	vector<int> strides;
	vector<int> pads;
	int output_c;
	int input_c;
	bool has_bias=0;
	bool has_relu=0;
	bool has_add=0;
	virtual Executer * clone() override
	{
		ConvImpl* r=new ConvImpl();
		r->algo=this->algo;
		return r;
	}
	virtual void init(Node &n) override
	{
		kernel_shape=n.params.at("kernel_shape").get_iarray();
		strides=n.params.at("strides").get_iarray();
		pads=n.params.at("pads").get_iarray();

		if(n.params.has_key("has_relu"))
		{ 
			has_relu=true;
			cudnnActivationMode_t mode=CUDNN_ACTIVATION_RELU;
			checkCUDNN(cudnnSetActivationDescriptor(acti_desc.descriptor, mode, CUDNN_NOT_PROPAGATE_NAN, 0.0));
		}
		else
		{
			cudnnActivationMode_t mode=CUDNN_ACTIVATION_IDENTITY;
			checkCUDNN(cudnnSetActivationDescriptor(acti_desc.descriptor, mode, CUDNN_NOT_PROPAGATE_NAN, 0.0));	
		}
		Engine &e= *this->engine;
		auto input_name=n.inputs.at("X");
		input.set_as_4d_tensor(e.tensors[input_name]);
		input_c=e.tensors[input_name].shape.dims[1];

		auto output_name=n.outputs.at("Y");
		output.set_as_4d_tensor(e.tensors[output_name]);
		output_c=e.tensors[output_name].shape.dims[1];

		if(n.inputs.find("B")!=n.inputs.end())
		{
			has_bias=1;
			auto bias_name=n.inputs.at("B");
			bias.set_as_4d_tensor_from_1d(e.tensors.at(bias_name));
		}

		if(n.inputs.find("add")!=n.inputs.end())
		{
			has_add=1;
			auto add_name=n.inputs.at("add");
			//if(e.tensors.at(add_name).strides_override.size()!=0) has_add=0;
			add.set_as_4d_tensor(e.tensors.at(add_name));
			
			op_type=CUDNN_OP_TENSOR_ADD;
			checkCUDNN(cudnnSetOpTensorDescriptor(op_desc.descriptor, op_type, CUDNN_DATATYPE,CUDNN_NOT_PROPAGATE_NAN));
		}


	

		auto weight_name=n.inputs.at("W");
		filter.data_ptr=e.tensors[weight_name].data_ptr;
		checkCUDNN(cudnnSetFilter4dDescriptor(filter.descriptor, CUDNN_DATATYPE,CUDNN_TENSOR_NCHW, output_c, input_c, kernel_shape[0], kernel_shape[1]));

		checkCUDNN(cudnnSetConvolution2dDescriptor(conv_desc.descriptor, pads[0], pads[1],
					strides[0], strides[1], 1/*dilationH*/, 1/*dilationW*/,

					CUDNN_CROSS_CORRELATION, CUDNN_DATATYPE));
	}
	virtual bool check_valid() override
	{
		const float alpha = 1.0f;
		const float beta = 0.0f;
		if(cudnnConvolutionForward(
					engine->cudnn_handle, &alpha, input.descriptor, input.data_ptr, filter.descriptor, filter.data_ptr,
					conv_desc.descriptor, algo, engine->cudnn_workspace->ptr, engine->cudnn_workspace->size,
					&beta, output.descriptor, output.data_ptr)!=CUDNN_STATUS_SUCCESS)
			return false;
		
		//no need
		/*     
		if(has_bias)
		{
			if(cudnnAddTensor(engine->cudnn_handle, &alpha, bias.descriptor, bias.data_ptr ,
						&alpha, output.descriptor, output.data_ptr)!=CUDNN_STATUS_SUCCESS)
				return false;
		}*/

		return true;
	}
	virtual void inference() override
	{
		const float alpha = 1.0f;
		const float beta = 0.0f;
		const float beta2 = 1.0f;

		//if(has_relu||algo==CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM)
		if(has_relu)
		{
			assert(has_bias);

			if(!has_add)
			{
				checkCUDNN(cudnnConvolutionBiasActivationForward(
				engine->cudnn_handle, &alpha, input.descriptor, input.data_ptr, filter.descriptor, filter.data_ptr,
				conv_desc.descriptor, algo, engine->cudnn_workspace->ptr, engine->cudnn_workspace->size,
				&beta, output.descriptor, output.data_ptr, bias.descriptor, bias.data_ptr, acti_desc.descriptor,
				output.descriptor, output.data_ptr));
			}
			else
			{
				checkCUDNN(cudnnConvolutionBiasActivationForward(
				engine->cudnn_handle, &alpha, input.descriptor, input.data_ptr, filter.descriptor, filter.data_ptr,
				conv_desc.descriptor, algo, engine->cudnn_workspace->ptr, engine->cudnn_workspace->size,
				&beta2, add.descriptor, add.data_ptr, bias.descriptor, bias.data_ptr, acti_desc.descriptor,
				output.descriptor, output.data_ptr));
			}

		}
		else
		{	
			checkCUDNN(cudnnConvolutionForward(
						engine->cudnn_handle, &alpha, input.descriptor, input.data_ptr, filter.descriptor, filter.data_ptr,
						conv_desc.descriptor, algo, engine->cudnn_workspace->ptr, engine->cudnn_workspace->size,
						&beta, output.descriptor, output.data_ptr));
			if(has_bias&&has_add)
			{	
					checkCUDNN(cudnnOpTensor(engine->cudnn_handle, op_desc.descriptor, &alpha, add.descriptor, add.data_ptr,
											&alpha, bias.descriptor, bias.data_ptr, &beta2, output.descriptor, output.data_ptr));	
			}
			else
			{
					if(has_bias)
					{
							checkCUDNN(cudnnAddTensor(engine->cudnn_handle, &alpha, bias.descriptor, bias.data_ptr ,
													&alpha, output.descriptor, output.data_ptr));
					}
					if(has_add)
					{
							checkCUDNN(cudnnAddTensor(engine->cudnn_handle, &alpha, add.descriptor, add.data_ptr ,
													&alpha, output.descriptor, output.data_ptr));
					}
			}
		}
	}
};

struct ReluImpl:Executer
{
	TensorDescriptor input;
	TensorDescriptor output;
	ActivationDescriptor acti_desc;

	virtual Executer * clone() override
	{
		ReluImpl* r=new ReluImpl();
		return r;
	}
	virtual void init(Node &n) override
	{
		Engine &e= *this->engine;

		auto input_name=n.inputs.at("X");
		auto output_name=n.outputs.at("Y");

		input.set_as_nd_tensor(e.tensors[input_name]);
		output.set_as_nd_tensor(e.tensors[output_name]);

		cudnnActivationMode_t mode=CUDNN_ACTIVATION_RELU;
		checkCUDNN(cudnnSetActivationDescriptor(acti_desc.descriptor, mode, CUDNN_NOT_PROPAGATE_NAN, 0.0));
	}
	virtual void inference()
	{
		const float alpha = 1.0f;
		const float beta = 0.0f;
		checkCUDNN(cudnnActivationForward(engine->cudnn_handle, acti_desc.descriptor, &alpha, input.descriptor, input.data_ptr, &beta, output.descriptor, output.data_ptr));
	}
};

struct IdentityImpl:Executer
{
	TensorDescriptor input;
	TensorDescriptor output;

	virtual Executer * clone() override
	{
		IdentityImpl* r=new IdentityImpl();
		return r;
	}
	virtual void init(Node &n) override
	{
		Engine &e= *this->engine;

		auto input_name=n.inputs.at("X");
		input.set_as_4d_tensor(e.tensors[input_name]);
		auto output_name=n.outputs.at("Y");
		output.set_as_4d_tensor(e.tensors[output_name]);

	}
	virtual void inference()
	{
		const float alpha = 1.0f;
		const float beta = 0.0f;
		extern int disable_icopy;
		//if(disable_icopy) {return;}
		if(input.data_ptr==output.data_ptr) {return ;}
		checkCUDNN(	cudnnTransformTensor(engine->cudnn_handle,&alpha,input.descriptor,
			input.data_ptr,&beta,output.descriptor,output.data_ptr));
	}
};

struct SoftmaxImpl:Executer
{
	TensorDescriptor input;
	TensorDescriptor output;

	cudnnSoftmaxAlgorithm_t algo;
	cudnnSoftmaxMode_t mode;
	virtual Executer * clone() override
	{
		SoftmaxImpl* r=new SoftmaxImpl();
		return r;
	}
	virtual void init(Node &n) override
	{
		Engine &e= *this->engine;
		algo=CUDNN_SOFTMAX_ACCURATE;
		mode=CUDNN_SOFTMAX_MODE_CHANNEL;
		auto input_name=n.inputs.at("X");
		input.set_as_nd_tensor(e.tensors[input_name]);
		auto output_name=n.outputs.at("Y");
		output.set_as_nd_tensor(e.tensors[output_name]);

	}
	virtual void inference()
	{
		const float alpha = 1.0f;
		const float beta = 0.0f;
		checkCUDNN(	cudnnSoftmaxForward(engine->cudnn_handle,algo,mode,&alpha,input.descriptor,
			input.data_ptr,&beta,output.descriptor,output.data_ptr));
	}
};

struct DropoutImpl:Executer
{
	TensorDescriptor input;
	TensorDescriptor output;
	float scale=0;

	virtual Executer * clone() override
	{
		DropoutImpl* r=new DropoutImpl();
		return r;
	}
	virtual void init(Node &n) override
	{
		Engine &e= *this->engine;

		auto input_name=n.inputs.at("X");
		auto output_name=n.outputs.at("Y");

		input.set_as_nd_tensor(e.tensors[input_name]);
		output.set_as_nd_tensor(e.tensors[output_name]);
		
		float ratio=n.params.at("ratio").get_float();
		scale=1/(1-ratio);
		scale=1;   // this line gives no diff with onnx runtime

	}
	virtual void inference()
	{
		//const float alpha = 1.0f;
		const float beta = 0.0f;
		checkCUDNN(	cudnnTransformTensor(engine->cudnn_handle,&scale,input.descriptor,
			input.data_ptr,&beta,output.descriptor,output.data_ptr));
	}
};


struct FlattenImpl:Executer
{
	TensorDescriptor input;
	TensorDescriptor output;

	virtual FlattenImpl * clone() override
	{
		FlattenImpl* r=new FlattenImpl();
		return r;
	}
	virtual void init(Node &n) override
	{
		Engine &e= *this->engine;

		auto input_name=n.inputs.at("X");
		input.set_as_2d_tensor_from_4d(e.tensors[input_name]);
		auto output_name=n.outputs.at("Y");
		output.set_as_nd_tensor(e.tensors[output_name]);

	}
	virtual void inference()
	{
		const float alpha = 1.0f;
		const float beta = 0.0f;
		checkCUDNN(	cudnnTransformTensor(engine->cudnn_handle,&alpha,input.descriptor,
			input.data_ptr,&beta,output.descriptor,output.data_ptr));
	}
};

struct ReshapeImpl:Executer
{
	TensorDescriptor input;
	TensorDescriptor output;

	virtual ReshapeImpl * clone() override
	{
		ReshapeImpl* r=new ReshapeImpl();
		return r;
	}
	virtual void init(Node &n) override
	{
		Engine &e= *this->engine;

		auto input_name=n.inputs.at("X");
		input.set_as_2d_tensor_from_4d(e.tensors[input_name]);
		auto output_name=n.outputs.at("Y");
		int h=e.tensors[input_name].shape.dims[2];
		int w=e.tensors[input_name].shape.dims[3];
		output.set_as_nd_tensor(e.tensors[output_name]);
		
		auto shape_name=n.inputs.at("shape");
		assert(e.tensors[shape_name].shape.size()==2);

	}
	virtual void inference()
	{
		const float alpha = 1.0f;
		const float beta = 0.0f;
		checkCUDNN(	cudnnTransformTensor(engine->cudnn_handle,&alpha,input.descriptor,
			input.data_ptr,&beta,output.descriptor,output.data_ptr));
	}
};

struct PoolImpl:Executer
{
	TensorDescriptor input;
	TensorDescriptor output;
	PoolingDescriptor pool_desc;
	vector<int> kernel_shape;
	vector<int> strides;
	vector<int> pads;
	string subtype;
	cudnnPoolingMode_t mode;

	virtual Executer * clone() override
	{
		PoolImpl* r=new PoolImpl();
		return r;
	}
	virtual void init(Node &n) override
	{
		kernel_shape=n.params.at("kernel_shape").get_iarray();
		strides=n.params.at("strides").get_iarray();
		pads=n.params.at("pads").get_iarray();
		subtype=n.params.at("subtype").get_string();
		
		Engine &e= *this->engine;

		auto input_name=n.inputs.at("X");
		input.set_as_4d_tensor(e.tensors[input_name]);

		auto output_name=n.outputs.at("Y");
		output.set_as_4d_tensor(e.tensors[output_name]);

		if(subtype=="AveragePool")
			mode=CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING;
		else if(subtype=="MaxPool")
			mode=CUDNN_POOLING_MAX;
		else assert(0==1);

		checkCUDNN(cudnnSetPooling2dDescriptor(pool_desc.descriptor, mode,CUDNN_PROPAGATE_NAN, kernel_shape[0], 
			kernel_shape[1], pads[0], pads[1], strides[0], strides[1]));

	}
	virtual void inference()
	{
		const float alpha = 1.0f;
		const float beta = 0.0f;
		checkCUDNN(cudnnPoolingForward(engine->cudnn_handle, pool_desc.descriptor,
			&alpha, input.descriptor, input.data_ptr,
			&beta, output.descriptor, output.data_ptr));
	}
};

struct AddImpl:Executer
{
	TensorDescriptor input1;
	TensorDescriptor input2;
	TensorDescriptor output;
	OpTensorDescriptor op_desc;
	cudnnOpTensorOp_t op_type;

	virtual Executer * clone() override
	{
		AddImpl* r=new AddImpl();
		return r;
	}
	virtual void init(Node &n) override
	{	
		Engine &e= *this->engine;

		auto input1_name=n.inputs.at("A");
		input1.set_as_nd_tensor(e.tensors[input1_name]);

		auto input2_name=n.inputs.at("B");
		input2.set_as_nd_tensor(e.tensors[input2_name]);

		auto output_name=n.outputs.at("C");
		output.set_as_nd_tensor(e.tensors[output_name]);

		op_type=CUDNN_OP_TENSOR_ADD;

		checkCUDNN(cudnnSetOpTensorDescriptor(op_desc.descriptor, op_type, CUDNN_DATATYPE,CUDNN_NOT_PROPAGATE_NAN));

	}
	virtual void inference()
	{
		const float alpha = 1.0f;
		const float beta = 0.0f;
		checkCUDNN(cudnnOpTensor(engine->cudnn_handle, op_desc.descriptor, &alpha, input1.descriptor, input1.data_ptr,
     		 &alpha, input2.descriptor, input2.data_ptr, &beta, output.descriptor, output.data_ptr));

	}
};

struct SplitImpl:Executer
{
	//TensorDescriptor input;
	deque<TensorDescriptor> sub_inputs;
	deque<TensorDescriptor> outputs;

	vector<int> split_sizes;
	vector<int> offsets;
	vector<int> skip;

	virtual Executer * clone() override
	{
		SplitImpl* r=new SplitImpl();
		return r;
	}
	virtual void init(Node &n) override
	{	
		Engine &e= *this->engine;
		split_sizes=n.params.at("sizes").get_iarray();
		offsets=n.params["offsets"].get_iarray();
		skip=vector<int>(offsets.size(),0);

		auto input_name=n.inputs.at("X");
		//input.set_as_4d_tensor(e.tensors[input_name]);

		sub_inputs.resize(split_sizes.size());
		outputs.resize(split_sizes.size());
		for(int i=0;i<(int)n.outputs.size();i++)
		{
			sub_inputs[i].set_as_4d_tensor_sub(e.tensors[input_name],offsets[i],split_sizes[i]);
			auto output_name=n.outputs.at(to_string(i));
			outputs[i].set_as_4d_tensor(e.tensors[output_name]);
			if(sub_inputs[i].data_ptr==outputs[i].data_ptr)
			{
				//printf("skip split\n");
				skip[i]=1;
			}
		}

	}
	virtual void inference()
	{
		const float alpha = 1.0f;
		const float beta = 0.0f;
		for(int i=0;i<(int)sub_inputs.size();i++)
		{
			extern int disable_copy;
			if(disable_copy) continue;
			if(skip[i]) continue;
			checkCUDNN(
			cudnnTransformTensor(engine->cudnn_handle,&alpha,sub_inputs[i].descriptor,
			sub_inputs[i].data_ptr,&beta,outputs[i].descriptor,outputs[i].data_ptr)
			);
		}
	}
};
struct ConcatImpl:Executer
{
	deque<TensorDescriptor> inputs;
	deque<TensorDescriptor> sub_outputs;
	//TensorDescriptor output;

	vector<int> concat_sizes;
	vector<int> offsets;
	vector<int> skip;

	virtual Executer * clone() override
	{
		ConcatImpl* r=new ConcatImpl();
		return r;
	}
	virtual void init(Node &n) override
	{	
		Engine &e= *this->engine;

		concat_sizes=n.params["sizes"].get_iarray();
		offsets=n.params["offsets"].get_iarray();
		skip=vector<int>(offsets.size(),0);

		auto output_name=n.outputs.at("Y");
		//output.set_as_4d_tensor(e.tensors[output_name]);

		sub_outputs.resize(concat_sizes.size());
		inputs.resize(concat_sizes.size());
		for(int i=0;i<(int)n.inputs.size();i++)
		{
			sub_outputs[i].set_as_4d_tensor_sub(e.tensors[output_name],offsets[i],concat_sizes[i]);
			auto input_name=n.inputs.at(to_string(i));
			inputs[i].set_as_4d_tensor(e.tensors[input_name]);
			if(sub_outputs[i].data_ptr==inputs[i].data_ptr)
			{
				//printf("skip concat\n");
				skip[i]=1;
			}
		}
	}
	virtual void inference()
	{
		const float alpha = 1.0f;
		const float beta = 0.0f;
		for(int i=0;i<(int)sub_outputs.size();i++)
		{
			extern int disable_copy;
			if(disable_copy) continue;
			if(skip[i]) continue;
			checkCUDNN(
							cudnnTransformTensor(engine->cudnn_handle,&alpha,inputs[i].descriptor,inputs[i].data_ptr,&beta,
									sub_outputs[i].descriptor,sub_outputs[i].data_ptr)
					  );
		}
	}
};
struct ConcatImpl2:Executer
{
	deque<TensorDescriptor> inputs;
	deque<TensorDescriptor> sub_outputs;
	//TensorDescriptor output;

	vector<int> concat_sizes;
	vector<int> offsets;
	vector<int> skip;

	int batch=-1;
	int hw=-1;
	int out_stride=-1;
	vector<int> in_stride;
	virtual Executer * clone() override
	{
		ConcatImpl2* r=new ConcatImpl2();
		return r;
	}
	virtual void init(Node &n) override
	{	
		Engine &e= *this->engine;

		concat_sizes=n.params["sizes"].get_iarray();
		offsets=n.params["offsets"].get_iarray();
		skip=vector<int>(offsets.size(),0);

		batch=n.get_output().shape.dims[0];
		hw=n.get_output().shape.dims[2]*n.get_output().shape.dims[3];

		auto output_name=n.outputs.at("Y");
		//output.set_as_4d_tensor(e.tensors[output_name]);
		out_stride=e.tensors[output_name].get_strides_or_override().at(0);

		sub_outputs.resize(concat_sizes.size());
		inputs.resize(concat_sizes.size());
		in_stride.resize(concat_sizes.size());
		for(int i=0;i<(int)n.inputs.size();i++)
		{
			sub_outputs[i].set_as_4d_tensor_sub(e.tensors[output_name],offsets[i],concat_sizes[i]);
			auto input_name=n.inputs.at(to_string(i));
			inputs[i].set_as_4d_tensor(e.tensors[input_name]);
			in_stride[i]=e.tensors[input_name].get_strides_or_override().at(0);
			if(sub_outputs[i].data_ptr==inputs[i].data_ptr)
			{
				//printf("skip concat\n");
				skip[i]=1;
			}
		}
	}
	virtual void inference()
	{
		const float alpha = 1.0f;
		const float beta = 0.0f;
		for(int i=0;i<(int)sub_outputs.size();i++)
		{
			extern int disable_copy;
			if(disable_copy) continue;
			if(skip[i]) continue;
			for(int j=0;j<batch;j++)
			cudaMemcpyAsync((DATATYPE*)sub_outputs[i].data_ptr+j*out_stride,(DATATYPE*)inputs[i].data_ptr+j*in_stride[i],hw*concat_sizes[i]*sizeof(DATATYPE),cudaMemcpyDeviceToDevice);
		}
	}
};
struct MatDescriptor:Descriptor
{
	DATATYPE *data_ptr=0;
	int num_of_rows=-1;
	int num_of_columns=-1;
	int leading_dimension=-1;

	void set_from_2d_tensor(DeviceTensor &device_tensor)
    {
        assert(data_ptr==0);
		auto &shape=device_tensor.shape;
		auto &stride_scale=device_tensor.force_stride_scale;
        data_ptr=(DATATYPE *)device_tensor.data_ptr;
        if(shape.dims.size()==2)
        {
			num_of_rows=shape.dims[0];
			num_of_columns=shape.dims[1];
			leading_dimension=num_of_columns*stride_scale;
			//leading_dimension=num_of_rows*stride_scale;
        }
        else
        {
            assert(0==1);
        }
    }
};
struct GemmImpl:Executer
{
	MatDescriptor matA;
	MatDescriptor matB;
	//MatDescriptor C;
	MatDescriptor matY;
	TensorDescriptor tensorC;
	TensorDescriptor tensorY;
	int m=-1,N=-1,k=-1;
	virtual GemmImpl * clone() override
	{
		GemmImpl* r=new GemmImpl();
		return r;
	}
	virtual void init(Node &n) override
	{
		Engine &e= *this->engine;
		assert(n.inputs.size()==3);

		auto A_name=n.inputs.at("A");
		matA.set_from_2d_tensor(e.tensors[A_name]);
		auto B_name=n.inputs.at("B");
		matB.set_from_2d_tensor(e.tensors[B_name]);

		auto C_name=n.inputs.at("C");
		tensorC.set_as_4d_tensor_from_1d(e.tensors[C_name]);//2d tensor is actually padded into 4d

		auto Y_name=n.outputs.at("Y");
		matY.set_from_2d_tensor(e.tensors[Y_name]);
		tensorY.set_as_nd_tensor(e.tensors[Y_name]);
		
		assert(matA.num_of_columns==matB.num_of_columns);
		assert(matA.num_of_rows==matY.num_of_rows);
		assert(matY.num_of_columns==matB.num_of_rows);
		m=matA.num_of_rows;
		k=matA.num_of_columns;
		N=matB.num_of_rows;
	}
	virtual void inference()
	{
		const float alpha = 1.0f;
		const float beta = 0.0f;
		//cublas assumes column-major 
		/*  //the first working version
		checkCUDA(cublasSgemm(engine->cublas_handle, CUBLAS_OP_T,CUBLAS_OP_N,
					N, m, k, &alpha, B.data_ptr, B.num_of_columns,A.data_ptr, A.num_of_columns, 
					&beta, Y.data_ptr, Y.num_of_columns));*/
		checkCUDA(cublasSgemm(engine->cublas_handle, CUBLAS_OP_T,CUBLAS_OP_N,
					N, m, k, &alpha, matB.data_ptr, matB.leading_dimension,matA.data_ptr, matA.leading_dimension, 
					&beta, matY.data_ptr, matY.leading_dimension));
		checkCUDNN(cudnnAddTensor(engine->cudnn_handle, &alpha, tensorC.descriptor,tensorC.data_ptr ,&alpha, tensorY.descriptor, tensorY.data_ptr));

	}
};

ExecuterManager::ExecuterManager()
{
	ConvImpl conv2d;
	register_exec("Conv","default",conv2d.clone());
	for(int i=1;i<8;i++)
	{
		//if(i!=1&&i!=5&&i!=6) continue;
		conv2d.algo=(cudnnConvolutionFwdAlgo_t)i;
		register_exec("Conv","algo_"+to_string(i),conv2d.clone());
	}
	register_exec("Relu","default",new ReluImpl);
	register_exec("Pool","default",new PoolImpl);
	register_exec("Add","default",new AddImpl);
	register_exec("Split","default",new SplitImpl);
	//register_exec("Concat","default",new ConcatImpl2);
	register_exec("Concat","default",new ConcatImpl);
	register_exec("Concat","memcpy",new ConcatImpl2);
	register_exec("Identity","default",new IdentityImpl);
	register_exec("Flatten","default",new FlattenImpl);
	register_exec("Gemm","default",new GemmImpl);
	register_exec("Reshape","default",new ReshapeImpl);
	register_exec("Dropout","default",new DropoutImpl);
	register_exec("Softmax","default",new SoftmaxImpl);
}
