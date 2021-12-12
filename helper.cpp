#include "helper.h"

#include <google/protobuf/util/json_util.h>
#include <google/protobuf/io/coded_stream.h>


struct ConvHelper:Helper
{
    ConvHelper()
    {
        type="Conv";
    }
    virtual void fill_inputs_outputs(onnx::NodeProto &pb_node,Node &node) override
    {
        assert(pb_node.input_size()==2||pb_node.input_size()==3);
        node.inputs["X"]=pb_node.input(0);
        node.inputs["W"]=pb_node.input(1);
        if(pb_node.input_size()==3)
          node.inputs["B"]=pb_node.input(2);
        node.outputs["Y"]=pb_node.output(0);
    }
    virtual void fill_attributes_post(onnx::NodeProto &pb_node,Node &node) override
    {
        auto &params=node.params;
        if(params.find("pads")==params.end())
        {
            vector<int> tmp(4,0);
            params["pads"].set_iarray(tmp);
        }
        else
        {
            vector<int> &tmp=params["pads"].get_iarray();
            if( tmp[0]!=tmp[2]||tmp[1]!=tmp[3] )
            {
                mylog(log_warn,"asymmetric padding %d %d %d %d\n",tmp[0],tmp[1],tmp[2],tmp[3]);
		short_pause();
                tmp[0]=max(tmp[0],tmp[2]);
                tmp[1]=max(tmp[1],tmp[3]);
            }
            //assert();
        }
    }
    virtual void shape_inference(Node &node) override
    {
        auto &params=node.params;

        auto &in_shape=node.get_input("X").shape;
        auto &weight_shape=node.get_input("W").shape;
        auto &out_shape=node.get_output("Y").shape;
        assert(in_shape.has_value() &&!out_shape.has_value());
        assert(in_shape.dims.size()==4);
        assert(weight_shape.dims.size()==4);
        out_shape.dims.resize(4);
        out_shape.dims[0]=in_shape.dims[0];
        out_shape.dims[1]=weight_shape.dims[0];
        out_shape.dims[2]=1+(in_shape.dims[2]-params["kernel_shape"].get_iarray()[0] +2*params["pads"].get_iarray()[0])/params["strides"].get_iarray()[0];
        out_shape.dims[3]=1+(in_shape.dims[3]-params["kernel_shape"].get_iarray()[1] +2*params["pads"].get_iarray()[1])/params["strides"].get_iarray()[1];

        //out_shape.dims[1]=node.params.at("")
    }

    virtual string get_key(Node &node)
    {
        string r;
        //assert(node.inputs.size()==1);
        auto &params=node.params;

        string &input_name=node.inputs.at("X");
        auto &input_tensor=node.graph->tensors[input_name];
        int output_c=node.get_input("W").shape.dims[0];
        int has_bias=(node.inputs.find("B")!=node.inputs.end());
		int has_relu=(int)params.has_key("has_relu");

        auto kernel_shape=params["kernel_shape"].get_iarray();
        auto pads=params["pads"].get_iarray();
		pads.resize(2);
        auto strides=params["strides"].get_iarray();
        r+=node.type;
        r+=","+input_tensor.shape.to_string();
        r+=",c"+to_string(output_c);
        r+=",b"+to_string(has_bias);
        r+=",r"+to_string(has_relu);
        r+=",k"+vec_to_string(kernel_shape);
        r+=",p"+vec_to_string(pads);
        r+=",s"+vec_to_string(strides);
        return r;
    }
};

struct PoolHelper:Helper
{
    PoolHelper()
    {
        type="Pool";
    }
    virtual void fill_inputs_outputs(onnx::NodeProto &pb_node,Node &node) override
    {
        assert(pb_node.input_size()==1);
        node.inputs["X"]=pb_node.input(0);
        node.outputs["Y"]=pb_node.output(0);
    }
    virtual void fill_attributes_post(onnx::NodeProto &pb_node,Node &node) override
    {
        auto &params=node.params;
        if(params.find("pads")==params.end())
        {
            vector<int> tmp(4,0);
            params["pads"].set_iarray(tmp);
        }
        else
        {
            vector<int> &tmp=params["pads"].get_iarray();
            if( tmp[0]!=tmp[2]||tmp[1]!=tmp[3] )
            {
                mylog(log_warn,"asymmetric padding %d %d %d %d\n",tmp[0],tmp[1],tmp[2],tmp[3]);
		        short_pause();
                tmp[0]=max(tmp[0],tmp[2]);
                tmp[1]=max(tmp[1],tmp[3]);
            }
        }
    }
    virtual void shape_inference(Node &node) override
    {
        auto &params=node.params;
        auto &in_shape=node.get_input("X").shape;
        auto &out_shape=node.get_output("Y").shape;
        auto &subtype=params.at("subtype").get_string();

        if(subtype=="GlobalAveragePool"||subtype=="GlobalMaxPool")   //has to put here, we don't have in_shapre before shape inference round
        {
            vector<int> kernel_shape(2,0);
            kernel_shape[0]=in_shape.dims[2];
            kernel_shape[1]=in_shape.dims[3];
            params["kernel_shape"].set_iarray(kernel_shape);
            vector<int> strides(2,1);
            params["strides"].set_iarray(strides);            
            if(subtype=="GlobalAveragePool")subtype="AveragePool";
            else subtype="MaxPool";

        }
        assert(in_shape.has_value() &&!out_shape.has_value());
        assert(in_shape.dims.size()==4);
        out_shape.dims.resize(4);
        out_shape.dims[0]=in_shape.dims[0];
        out_shape.dims[1]=in_shape.dims[1];;
        out_shape.dims[2]=1+(in_shape.dims[2]-params["kernel_shape"].get_iarray()[0]+2*params["pads"].get_iarray()[0])/params["strides"].get_iarray()[0];
        out_shape.dims[3]=1+(in_shape.dims[3]-params["kernel_shape"].get_iarray()[1]+2*params["pads"].get_iarray()[1])/params["strides"].get_iarray()[1];

        //out_shape.dims[1]=node.params.at("")
    }

    virtual string get_key(Node &node)
    {
        string r;
        assert(node.inputs.size()==1);
        auto &params=node.params;

        string &input_name=node.inputs.begin()->second;
        auto &input_tensor=node.graph->tensors[input_name];

        auto kernel_shape=params["kernel_shape"].get_iarray();
        auto pads=params["pads"].get_iarray();
		pads.resize(2);
        auto strides=params["strides"].get_iarray();
        r+=node.type;
        r+=","+input_tensor.shape.to_string();
        r+=",k"+vec_to_string(kernel_shape);
        r+=",p"+vec_to_string(pads);
        r+=",s"+vec_to_string(strides);
        return r;
    }
};

struct ReluHelper:Helper
{
    ReluHelper()
    {
        type="Relu";
    }
    virtual void fill_inputs_outputs(onnx::NodeProto &pb_node,Node &node) override
    {
        assert(pb_node.input_size()==1);
        node.inputs["X"]=pb_node.input(0);
        node.outputs["Y"]=pb_node.output(0);
    }
};

struct IdentityHelper:Helper
{
    IdentityHelper()
    {
        type="Identity";
    }
    virtual void fill_inputs_outputs(onnx::NodeProto &pb_node,Node &node) override
    {
        assert(pb_node.input_size()==1);
        node.inputs["X"]=pb_node.input(0);
        node.outputs["Y"]=pb_node.output(0);
    }
};

struct SoftmaxHelper:Helper
{
    SoftmaxHelper()
    {
        type="Softmax";
    }
    virtual void fill_inputs_outputs(onnx::NodeProto &pb_node,Node &node) override
    {
        assert(pb_node.input_size()==1);
        node.inputs["X"]=pb_node.input(0);
        node.outputs["Y"]=pb_node.output(0);
    }
};


struct DropoutHelper:Helper
{
    DropoutHelper()
    {
        type="Dropout";
    }
    virtual void fill_inputs_outputs(onnx::NodeProto &pb_node,Node &node) override
    {
        assert(pb_node.input_size()==1);
        node.inputs["X"]=pb_node.input(0);
        node.outputs["Y"]=pb_node.output(0);
    }
    virtual void fill_attributes(onnx::NodeProto &pb_node,Node &node) override
    {
        Helper::fill_attributes(pb_node,node);
        if(!node.params.has_key("ratio"))
        {
            node.params["ratio"].set_float(0.5);
        }
    }
};

struct ReshapeHelper:Helper
{
    ReshapeHelper()
    {
        type="Reshape";
    }
    virtual void fill_inputs_outputs(onnx::NodeProto &pb_node,Node &node) override
    {
        assert(pb_node.input_size()==2);
        node.inputs["X"]=pb_node.input(0);
        node.inputs["shape"]=pb_node.input(1);
        node.outputs["Y"]=pb_node.output(0);
    }
    virtual void shape_inference(Node &node) override
    {
        auto &out_shape=node.get_output("Y").shape;
        auto &in_shape=node.get_input("X").shape;
        assert(in_shape.has_value() &&!out_shape.has_value());
        assert(in_shape.dims.size()==4);
        out_shape.dims.resize(2);
        out_shape.dims[0]=in_shape.dims[0];
        out_shape.dims[1]=1;
        for(int i=1;i<=3;i++)
            out_shape.dims[1]*=in_shape.dims[i];
    }
};

struct AddHelper:Helper
{
    AddHelper()
    {
        type="Add";
    }
    virtual void fill_inputs_outputs(onnx::NodeProto &pb_node,Node &node) override
    {
        assert(pb_node.input_size()==2);
        node.inputs["A"]=pb_node.input(0);
        node.inputs["B"]=pb_node.input(1);
        node.outputs["C"]=pb_node.output(0);
    }
    virtual void shape_inference(Node &node) override
    {
        auto &params=node.params;
        auto &in_shape1=node.get_input("A").shape;
        auto &in_shape2=node.get_input("B").shape;
        auto &out_shape=node.get_output("C").shape;
        assert(in_shape1.has_value() &&!out_shape.has_value());
        assert(in_shape1.dims==in_shape2.dims);
        out_shape.dims=in_shape1.dims;
    }
};

struct ConcatHelper:Helper
{
    ConcatHelper()
    {
        type="Concat";
    }
    virtual void fill_inputs_outputs(onnx::NodeProto &pb_node,Node &node) override
    {
        assert(pb_node.input_size()>=2);
        assert(pb_node.output_size()==1);
        for(int i=0;i<pb_node.input_size();i++)
        {
            node.inputs[to_string(i)]=pb_node.input(i);
        }
        node.outputs["Y"]=pb_node.output(0);
    }

    virtual void shape_inference(Node &node) override
    {
        auto &out_shape=node.get_output("Y").shape;
        auto &in_shape=node.get_input("0").shape;
        assert(in_shape.has_value() &&!out_shape.has_value());
        out_shape.dims=in_shape.dims;
        for(int i=1;i<(int)node.inputs.size();i++)
        {
            out_shape.dims[1]+=node.get_input(to_string(i)).shape.dims[1];
        }

        vector<int> concat_sizes;   // has to put it here since we need shape
		vector<int> offsets;
		concat_sizes.resize(node.inputs.size());
		offsets.resize(node.inputs.size());
		int sum=0;
		for(int i=0;i<(int)node.inputs.size();i++)
		{
			concat_sizes[i]=node.get_input(to_string(i)).shape.dims[1];
			offsets[i]=sum;
			sum+=concat_sizes[i];
		}
        node.params["sizes"].set_iarray(concat_sizes);
        node.params["offsets"].set_iarray(offsets);
    }
    virtual string get_key(Node &node)
    {
        string r;
        r+="Concat";
        auto &out_shape=node.get_output("Y").shape;
        assert(out_shape.dims.size()==4);
        vector<int> new_shape;
        new_shape.push_back(out_shape.dims[0]);
        new_shape.push_back(out_shape.dims[2]);
        new_shape.push_back(out_shape.dims[3]);
        vector<int> channels;
        auto &out_node=node.get_output("Y");
        for(int i=0;i<(int)node.inputs.size();i++)
        {
            auto &in_node=node.get_input(to_string(i));
            if(in_node.mem.part_of!=out_node.name)
            {
                channels.push_back(in_node.shape.dims[1]);
            }
        }
		if(channels.empty())
		{
			r+=",empty";
			return r;
		}
        r+=","+vec_to_string(new_shape);
        r+=","+vec_to_string(channels);
        return r;

    }
};

struct FlattenHelper:Helper
{
    FlattenHelper()
    {
        type="Flatten";
    }
    virtual void fill_inputs_outputs(onnx::NodeProto &pb_node,Node &node) override
    {
        assert(pb_node.input_size()==1);
        assert(pb_node.output_size()==1);
        node.inputs["X"]=pb_node.input(0);
        node.outputs["Y"]=pb_node.output(0);
    }
    virtual void shape_inference(Node &node) override
    {
        auto &out_shape=node.get_output("Y").shape;
        auto &in_shape=node.get_input("X").shape;
        assert(in_shape.has_value() &&!out_shape.has_value());
        assert(in_shape.dims.size()==4);
        out_shape.dims.resize(2);
        out_shape.dims[0]=in_shape.dims[0];
        out_shape.dims[1]=1;
        for(int i=1;i<=3;i++)
            out_shape.dims[1]*=in_shape.dims[i];
    }
};
struct GemmHelper:Helper
{
    GemmHelper()
    {
        type="Gemm";
    }
    virtual void fill_inputs_outputs(onnx::NodeProto &pb_node,Node &node) override
    {
        //assert(pb_node.input_size()==2||pb_node.input_size()==3);
        assert(pb_node.input_size()==3); //seems like it must be 3
        node.inputs["A"]=pb_node.input(0);
        node.inputs["B"]=pb_node.input(1);
        if(pb_node.input_size()==3)
        {
            node.inputs["C"]=pb_node.input(2);
        }
        node.outputs["Y"]=pb_node.output(0);
    }
    virtual void shape_inference(Node &node) override
    {
        auto &params=node.params;
        assert(params.has_key("transB"));
        assert(params["transB"].get_int()==1);
        if(params.has_key("transA"))
        {
            assert(params["transA"].get_int()==0);
        }

        auto &in_shape1=node.get_input("A").shape;
        auto &in_shape2=node.get_input("B").shape;
        auto &out_shape=node.get_output("Y").shape;
        assert(in_shape1.dims.size()==2);
        assert(in_shape2.dims.size()==2);
        assert(in_shape1.dims[1]==in_shape2.dims[1]);
        assert(in_shape1.has_value() &&!out_shape.has_value());
        out_shape.dims.resize(2);
        out_shape.dims[0]=in_shape1.dims[0];
        out_shape.dims[1]=in_shape2.dims[0];
    }

    virtual string get_key(Node &node)
    {
        string r;
        auto &in_shape1=node.get_input("A").shape;
        auto &in_shape2=node.get_input("B").shape;
        r+="Gemm";
        r+=","+in_shape1.to_string();
        r+=","+in_shape2.to_string();
        return r;
    }
};

struct SplitHelper:Helper
{
    SplitHelper()
    {
        type="Split";
    }
    virtual void fill_inputs_outputs(onnx::NodeProto &pb_node,Node &node) override
    {
        assert(pb_node.input_size()==1);
        node.inputs["X"]=pb_node.input(0);
        for(int i=0;i<pb_node.output_size();i++)
        {
            node.outputs[to_string(i)]=pb_node.output(i);
        }
    }
    virtual void fill_attributes_post(onnx::NodeProto &pb_node,Node &node) override
    {
        vector<int> split_sizes=node.params.at("split").get_iarray();
        vector<int> offsets;
		int sum=0;
		offsets.resize(split_sizes.size());
		for(int i=0;i<(int)split_sizes.size();i++)
		{
			offsets[i]=sum;
			sum+=split_sizes[i];
		}
        node.params.erase("split");
        node.params["offsets"].set_iarray(offsets);
        node.params["sizes"].set_iarray(split_sizes);
    }
    virtual void shape_inference(Node &node) override
    {
       // auto &out_shape=node.get_output("Y").shape;
        assert(node.outputs.size()==node.params["sizes"].get_iarray().size());
        auto &in_shape=node.get_input("X").shape;
        assert(in_shape.has_value());
        for(int i=0;i<(int)node.outputs.size();i++)
        {
            assert(!node.get_output(to_string(i)).shape.has_value());
        }
        int sum=0;
        for(int i=0;i<(int)node.params["sizes"].get_iarray().size();i++)
        {
            sum+=node.params["sizes"].get_iarray()[i];
        }
        assert(sum==in_shape.dims[1]);
        for(int i=0;i<(int)node.outputs.size();i++)
        {
            auto &out_shape=node.get_output(to_string(i)).shape;
            out_shape.dims=in_shape.dims;
            out_shape.dims[1]=node.params["sizes"].get_iarray()[i];

        }
    }
    virtual string get_key(Node &node)
    {
        string r;
        r+="Split";
        auto &in_shape=node.get_input("X").shape;
        assert(in_shape.dims.size()==4);
        vector<int> new_shape;
        new_shape.push_back(in_shape.dims[0]);
        new_shape.push_back(in_shape.dims[2]);
        new_shape.push_back(in_shape.dims[3]);
        vector<int> channels;
        auto &in_node=node.get_input("X");
        for(int i=0;i<(int)node.outputs.size();i++)
        {
            auto &out_node=node.get_output(to_string(i));
            if(out_node.mem.part_of!=in_node.name)
            {
                channels.push_back(out_node.shape.dims[1]);
            }
        }
		if(channels.empty())
		{
			r+=",empty";
			return r;
		}
        r+=","+vec_to_string(new_shape);
        r+=","+vec_to_string(channels);
        return r;

    }
};


    HelperManager::HelperManager()
    {
        register_helper(new Helper);
        register_helper(new ConvHelper);
        register_helper(new ReluHelper);
        register_helper(new AddHelper);
        register_helper(new ConcatHelper);
        register_helper(new FlattenHelper);
        register_helper(new GemmHelper);
        register_helper(new SplitHelper);
        register_helper(new PoolHelper);
        register_helper(new IdentityHelper);
        register_helper(new DropoutHelper);
        register_helper(new ReshapeHelper);
        register_helper(new SoftmaxHelper);
    }
    HelperManager *HelperManager:: get_instance()
    {
        if(p==0) p=new HelperManager;
        return p;
    }
    void HelperManager::register_helper(Helper *helper)
    {
        assert(mp.find(helper->type)==mp.end());
        mp[helper->type]=helper;
    }
    Helper *HelperManager::get_helper(string type)
    {
        if(mp.find(type)==mp.end())
        {
            mylog(log_warn,"Could not find Helper for %s, using default\n",type.c_str());
	    short_pause();
            return mp["Default"];
        }
        else return mp[type];
    }

HelperManager * HelperManager::p=0;

map<string, string> node_remap=
{
    {"AveragePool","Pool"},
    {"MaxPool","Pool"},
    {"GlobalAveragePool","Pool"},
    {"GlobalMaxPool","Pool"}
};

string rename_tensor(string name)
{
    return "tensor:"+name;
}
void Graph::from_onnx(string file_name)
{

  std::ifstream input(file_name, std::ios::ate | std::ios::binary); // open file and move current position in file to the end

  std::streamsize size = input.tellg(); // get current position in file
  input.seekg(0, std::ios::beg); // move to start of file

  std::vector<char> buffer(size);
  input.read(buffer.data(), size); // read raw data

  onnx::ModelProto model;
  google::protobuf::io::CodedInputStream code_input_stream((unsigned char *)buffer.data(), size);

#if GOOGLE_PROTOBUF_VERSION < 3011000
  code_input_stream.SetTotalBytesLimit(2000*1024*1024,1000*1024*1024);
#else
  code_input_stream.SetTotalBytesLimit(2000*1024*1024);
#endif


  //model.ParseFromArray(buffer.data(), size); // parse protobuf
  model.ParseFromCodedStream(&code_input_stream);

  auto &in_graph = model.graph();
  auto iv=model.ir_version();
  cout<<"ir_version="<<iv<<endl;
  cout<<"size="<<model.opset_import_size()<<endl;
  std::cout<<"opset.version="<<model.opset_import(0).version()<<std::endl;

    auto & pb_graph=in_graph;
    assert(nodes.size()==0);
    assert(tensors.size()==0);
    auto &helper_manager=*HelperManager::get_instance();
	int num = pb_graph.node_size();
	for (int i = 0; i < num; i++)    //遍历每个node结构
	{
		Node node;
		onnx::NodeProto pb_node = pb_graph.node(i);
		std::string node_name = pb_node.name();
		if(node_name=="")
		{
			node_name=workspace->gen_node_name();
		}
		node.name=node_name;
		const ::google::protobuf::RepeatedPtrField< ::onnx::AttributeProto> attr = pb_node.attribute();        //每个node结构的参数信息
		const std::string type = pb_node.op_type();
        if(node_remap.find(type)==node_remap.end())
        {
		    node.type=type;
        }
        else
        {
            node.type=node_remap[type];
            node.params["subtype"].set_string(type);
        }
		int in_size = pb_node.input_size();
		int out_size = pb_node.output_size();

		Helper* helper=helper_manager.get_helper(node.type);
		helper->fill_node(pb_node,node);
		nodes[node.name]=node;
	}
    for(auto &x :nodes)
    {
        for(auto &y:x.second.inputs)
        {
            if(nodes.find(y.second)!=nodes.end())//special case: tensor has same name as node
            {
                y.second=rename_tensor(y.second);
            }
        }
        for(auto &y:x.second.outputs)
        {
            if(nodes.find(y.second)!=nodes.end())
            {
                y.second=rename_tensor(y.second);
            }
        }
    }
	//cout<<in_graph.DebugString()<<endl;
	for (int i = 0; i < pb_graph.initializer_size(); i++) 
	{
		auto &pb_init=pb_graph.initializer(i);
		Init init;
        string name=pb_init.name();
        if(nodes.find(name)!=nodes.end())
        {
            name=rename_tensor(name);
        }
		init.name=name;
		int size=1;
		for(int j=0; j<pb_init.dims_size();j++)
		{
			init.shape.dims.push_back(pb_init.dims(j));
			size*=pb_init.dims(j);
		}
		init.data.resize(size);
		if(pb_init.raw_data().size()==size*sizeof(DATATYPE))
		{
			memcpy(init.data.data(),pb_init.raw_data().c_str(),pb_init.raw_data().size());
		}
		else if(pb_init.float_data_size()==size)
		{
			for(int i=0;i<size;i++)
				init.data[i]=pb_init.float_data(i);
			//memcpy(init.data.data(),pb_init.float_data().c_str(,pb_init.raw_data().size());
		}
		else 
		{
			mylog(log_warn,"can't recongize initializers for %s\n",init.name.c_str());
			short_pause();
			//continue;
			//assert(0==1);
		}

		workspace->inits[init.name]=init;
	}

	for (int i = 0; i < pb_graph.input_size(); i++) 
	{
		auto &a=pb_graph.input(i);
		string name=a.name();
        if(nodes.find(name)!=nodes.end())
        {
            name=rename_tensor(name);
        }
        if(workspace->inits.find(name)!=workspace->inits.end())
        {
            continue;//old version tread inits as inputs
        }
		auto &b=a.type().tensor_type().shape();
		Shape shape;
		for(int j=0; j<b.dim_size();j++)
		{
			shape.dims.push_back(b.dim(j).dim_value());
		}
		inputs[name]=shape;
	}
	for (int i = 0; i < pb_graph.output_size(); i++) 
	{
		auto &a=pb_graph.output(i);
		string name=a.name();
        if(nodes.find(name)!=nodes.end())
        {
            name=rename_tensor(name);
        }
		auto &b=a.type().tensor_type().shape();
		Shape shape;
		for(int j=0; j<b.dim_size();j++)
		{
			shape.dims.push_back(b.dim(j).dim_value());
		}
		outputs[name]=shape;
	}
    post_process();
}

void Graph::save_model(string file_name)
{
    post_process_preserve();
    resolve_lazy();
    post_process_preserve();

    model_file::Graph pb_graph;
    pb_graph.set_comment(comment);
    for(auto &x:nodes)
    {
        auto &node=x.second;
        auto &pb_node= *pb_graph.add_nodes();
        pb_node.set_name(node.name);
        pb_node.set_type(node.type);
        for(auto &y:node.params)
        {
            auto &value=y.second;
            auto &pb_param=*pb_node.add_params();
            pb_param.set_name(y.first);
            switch(value.type)
            {
                case Value::Type::NONE:
                    pb_param.set_type(model_file::Type::UNDEFINED);
                    break;
                case Value::Type::FLOAT:
                    pb_param.set_type(model_file::Type::FLOAT);
                    pb_param.set_f(value.get_float());
                    break;
                case Value::Type::IARRAY:
                    pb_param.set_type(model_file::Type::INTS);
                    {
                        auto vec=value.get_iarray();
                        for(auto &z:vec)
                            pb_param.add_ints(z);
                    }
                    break;
                case Value::Type::STRING:
                    pb_param.set_type(model_file::Type::STRING);
                    pb_param.set_s(value.get_string());
                    break;
                case Value::Type::INT:
                    pb_param.set_type(model_file::Type::INT);
                    pb_param.set_i(value.get_int());
                    break;
                default:
                    assert(0==1);
            }
        }
        for(auto &y:node.inputs)
        {
            auto &pb_input=*pb_node.add_inputs();
            pb_input.set_idx(y.first);
            pb_input.set_name(y.second);    
        }

        for(auto &y:node.outputs)
        {
            auto &pb_output=*pb_node.add_outputs();
            pb_output.set_idx(y.first);
            pb_output.set_name(y.second);    
        }
        pb_node.set_impl_name(node.extra.algo_name);
    }

    for(auto &y:tensors)
    {
        auto &tensor=y.second;
        if(!tensor.is_input() && !tensor.is_output() &&!tensor.is_constant())
        {
            //continue;
        }
        auto &pb_tensor= *pb_graph.add_tensors();
        pb_tensor.set_is_input(tensor.is_input());
        pb_tensor.set_is_output(tensor.is_output());
        pb_tensor.set_name(tensor.name);
        for(auto &z:tensor.shape.dims)
            pb_tensor.add_shape(z);
        if(tensor.is_constant())
        {
            auto &init= workspace->inits.at(tensor.name);
			google::protobuf::RepeatedField<DATATYPE> data(init.data.begin(), init.data.end());
			pb_tensor.mutable_data()->Swap(&data);
            /*for(auto &z:init.data)
            {
                pb_tensor.add_data(z);
            }*/
        }
        pb_tensor.set_part_of(tensor.mem.part_of);
        pb_tensor.set_offset(tensor.mem.offset);
    }
    fstream output(file_name, ios::out | ios::trunc | ios::binary);
    if (!pb_graph.SerializeToOstream(&output)) {
        mylog(log_fatal,"failed to write file %s\n",file_name.c_str());
    }
    return ;
}

void Graph::load_model(string file_name)
{
  std::ifstream input(file_name, std::ios::ate | std::ios::binary); // open file and move current position in file to the end

  std::streamsize size = input.tellg(); // get current position in file
  input.seekg(0, std::ios::beg); // move to start of file

  std::vector<char> buffer(size);
  input.read(buffer.data(), size); // read raw data

  model_file::Graph pb_graph;
  google::protobuf::io::CodedInputStream code_input_stream((unsigned char *)buffer.data(), size);

#if GOOGLE_PROTOBUF_VERSION < 3011000
  code_input_stream.SetTotalBytesLimit(2000*1024*1024,1000*1024*1024);
#else
  code_input_stream.SetTotalBytesLimit(2000*1024*1024);
#endif

  pb_graph.ParseFromCodedStream(&code_input_stream);

  for(int i=0;i<pb_graph.nodes_size();i++)
  {
      Node node;
      model_file::Node pb_node=pb_graph.nodes(i);
      node.name=pb_node.name();
      node.type=pb_node.type();
      for(int j=0;j<pb_node.params_size();j++)
      {
          Value value;
          model_file::Param pb_param=pb_node.params(j);
          switch(pb_param.type())
          {
            case model_file::Type::UNDEFINED:
                break;
            case model_file::Type::FLOAT:
                value.set_float(pb_param.f());
                break;
            case model_file::Type::STRING:
                value.set_string(pb_param.s());
                break;
            case model_file::Type::INT:
                value.set_int(pb_param.i());
                break;
            case model_file::Type::INTS:
                {
                    vector<int> vec;
                    for(int k=0;k<pb_param.ints_size();k++)
                        vec.push_back(pb_param.ints(k));
                    value.set_iarray(vec);
                }
                break;
            default:
                assert(0==1);
          }
          node.params[pb_param.name()]=value;
      }
      for(int j=0;j<pb_node.inputs_size();j++)
      {
          node.inputs[pb_node.inputs(j).idx()]=pb_node.inputs(j).name();
      }
      for(int j=0;j<pb_node.outputs_size();j++)
      {
          //printf("<%s>\n",pb_node.outputs(j).name().c_str());
          node.outputs[pb_node.outputs(j).idx()]=pb_node.outputs(j).name();
      }
      node.extra.algo_name=pb_node.impl_name();
      nodes[node.name]=node;
  }
  vector<string> exist_inits;
  for(int i=0;i<pb_graph.tensors_size();i++)
  {
      model_file::Tensor pb_tensor=pb_graph.tensors(i);
      Shape shape;
      for(int j=0;j<pb_tensor.shape_size();j++)
      {
          shape.dims.push_back(pb_tensor.shape(j));
      }
      string name=pb_tensor.name();
      if(pb_tensor.is_input()) inputs[name]=shape;
      if(pb_tensor.is_output()) outputs[name]=shape;
      if(pb_tensor.data_size()!=0)
      {
          vector<DATATYPE> vec(pb_tensor.data_size());
		  memcpy(vec.data(), pb_tensor.data().data(),pb_tensor.data_size()*sizeof(DATATYPE));
          if(workspace->inits.find(name)!=workspace->inits.end())
          {
              exist_inits.push_back(name);
              //
          }
          auto &init=workspace->inits[name];
          init.data=vec;
          init.name=name;
          init.shape=shape;
      }
	  tensors[name].mem.part_of=pb_tensor.part_of();
	  tensors[name].mem.offset=pb_tensor.offset();
  }
  if(exist_inits.size())
  {
        mylog(log_warn,"tensor name exists in workspace: ");
        for(auto &x:exist_inits)
        {
            log_bare(log_warn,"%s ",x.c_str());
        }
        log_bare(log_warn,"\n");
  }
  post_process_preserve();

}
void Graph::shape_inference()
{
    auto &helper_manager=* HelperManager::get_instance();
    for(auto x:topo_order)
    {
        auto &node=nodes[x];
        string type=node.type;
        Helper* helper=helper_manager.get_helper(type);
        helper->shape_inference(node);
    }
}
