#include "trans.h"
#include "profiler.h"
void delete_node(Graph &graph,const string& tensor_name, const string &node_name)
{
    auto &tensor=graph.tensors.at(tensor_name);
    auto &node=graph.nodes.at(node_name);

    assert(tensor.write_by.name!="<init>");
    assert(tensor.write_by.idx!="");
    auto &pre_node=graph.nodes.at(tensor.write_by.name);
    assert(node.outputs.size()==1);
    pre_node.outputs.at(tensor.write_by.idx)=node.outputs.begin()->second;

    /*
    for(auto &x: node.get_output().read_by)
    {
        graph.nodes[x.name].inputs[x.idx]=tensor.write_by.name;
    }*/

    graph.nodes.erase(node.name);
}
vector<DATATYPE> concat_vec(const vector<DATATYPE> &v1,const vector<DATATYPE>&v2 )
{
    vector<DATATYPE>r=v1;
    r.insert(r.end(),v2.begin(),v2.end());
    return r;
}

vector<DATATYPE> enlarge_vec(const vector<DATATYPE> &v,int h,int w, int new_h,int new_w,int pad_h,int pad_w)
{
    //printf("enlarge (%d,%d) (%d,%d)\n", h,w,new_h,new_w);
    assert(v.size()%(h*w)==0);
    int repeats=v.size()/(h*w);
    vector<DATATYPE> r(repeats*new_h*new_w,0);
    for(int x=0;x<repeats;x++)
    {
        DATATYPE (*src)[w];
        src= (decltype(src))(v.data()+x*h*w);
            
        DATATYPE (*dst)[new_w];
        dst= (decltype(dst))(r.data()+x*new_h*new_w);

        for(int i=0;i<h;i++)
            for(int j=0;j<w;j++)
            {
                dst[i+pad_h][j+pad_w]=src[i][j];
            }
    }
    return r;
}
bool enlarge_conv(Graph &graph,const string &node_name1, const vector<int> &ker)
{
    assert(ker.size()==2);
    auto & node1=graph.nodes.at(node_name1);
    assert(node1.type==CONV);
    if(node1.inputs.find("B")==node1.inputs.end()) return 0;
    if(node1.inputs.find("add")!=node1.inputs.end()) return 0;
    auto &ker1=node1.params.at("kernel_shape").get_iarray();
    auto old_ker1=ker1;
    if(ker1==ker) return 0;

    for(int i=0;i<2;i++)
    {
        int pad_change=(ker[i]-ker1[i])/2;
        auto &pads=node1.params.at("pads").get_iarray();
        pads[i]+=pad_change;
        pads[i+2]+=pad_change; 
    }
    ker1=ker;
    string weight_tensor_name=graph.workspace->gen_var_name();
    
    auto &shape1= node1.get_input("W").shape;

    Init & weight_init=graph.workspace->inits[weight_tensor_name];
    weight_init.name=weight_tensor_name;
    weight_init.shape=shape1;
    weight_init.shape.dims.at(2)=ker.at(0);
    weight_init.shape.dims.at(3)=ker.at(1);

    auto &inits=graph.workspace->inits;

    string &w_name=node1.inputs.at("W");
    weight_init.lazy.has_value=1;
    weight_init.lazy.deps.push_back(w_name);
    weight_init.lazy.func=[w_name,old_ker1,ker,&inits]()
    {
		int pad_h=(ker[0]-old_ker1[0])/2;
		int pad_w=(ker[1]-old_ker1[1])/2;
        return enlarge_vec( inits.at(w_name).data,old_ker1[0],old_ker1[1],ker[0],ker[1],pad_h,pad_w);
    };
    node1.inputs.at("W")=weight_tensor_name;
    graph.post_process();
    return 1;
}
bool merge_conv_neq(Graph &graph, const string &node_name1,const string &node_name2)
{
    //printf("trying to merege %s %s neq\n",node_name1.c_str(),node_name2.c_str());
    auto & node1=graph.nodes.at(node_name1);
    auto & node2=graph.nodes.at(node_name2);
    assert(node1.type==CONV);
    assert(node2.type==CONV);
    assert(node1.get_input("X").name==node2.get_input("X").name);
    //auto & input_tensor=node1.get_input("X");
    if(node1.inputs.find("B")==node1.inputs.end()) return 0;
    if(node2.inputs.find("B")==node2.inputs.end()) return 0;
    if(node1.inputs.find("add")!=node1.inputs.end()) return 0;
    if(node2.inputs.find("add")!=node2.inputs.end()) return 0;
    //if(node1.params.has_key("has_relu")!=node2.params.has_key("has_relu")) return 0;
    if(node1.params.at("strides").get_iarray()!=node2.params.at("strides").get_iarray()) return 0;

    if(node1.get_output().shape.dims.at(2)!=node2.get_output().shape.dims.at(2)) return 0;
    if(node1.get_output().shape.dims.at(3)!=node2.get_output().shape.dims.at(3)) return 0;

    auto &ker1=node1.params.at("kernel_shape").get_iarray();
    auto &ker2=node2.params.at("kernel_shape").get_iarray();
    if(ker1==ker2) return 0;

    vector<int> ker(2);
    for(int i=0;i<2;i++)
    {
        ker[i]=max(ker1[i],ker2[i]);
        if(ker[i]>3) return 0;
        if((ker[i]-ker1[i])%2!=0) return 0;
        if((ker[i]-ker2[i])%2!=0) return 0;
    }

    if(ker1!=ker)
    {
        assert(enlarge_conv(graph,node_name1,ker)==1);
    }
    if(ker2!=ker)
    {
        assert(enlarge_conv(graph,node_name2,ker)==1);
    }

    int r=merge_conv(graph, node_name1,node_name2);
    assert(r==1);
    //printf("merge neq success\n");
 
    return r;
}
bool release_relu(Graph &graph,const string &node_name)
{
    auto & node=graph.nodes.at(node_name);
    if(!node.params.has_key("has_relu")) return 0;

    node.params.erase("has_relu");

    string relu_name=graph.workspace->gen_node_name();
    assert(graph.tensors.find(relu_name)==graph.tensors.end());
    Node &relu=graph.nodes[relu_name];
    relu.name=relu_name;
    relu.type=RELU;
    relu.outputs["Y"]=node.outputs.at("Y");
    relu.inputs["X"]=node.name+":0";

    node.outputs.at("Y")=node.name+":0";
	graph.post_process();
    return 1;
}

bool merge_conv(Graph &graph, const string &node_name1,const string &node_name2)
{
    //printf("trying to merege %s %s\n",node_name1.c_str(),node_name2.c_str());
    auto & node1=graph.nodes.at(node_name1);
    auto & node2=graph.nodes.at(node_name2);
    assert(node1.type==CONV);
    assert(node2.type==CONV);
    assert(node1.get_input("X").name==node2.get_input("X").name);
    //auto & input_tensor=node1.get_input("X");                   //tensor referece will be invalid after fill_extra()
    if(node1.inputs.find("B")==node1.inputs.end()) return 0;
    if(node2.inputs.find("B")==node2.inputs.end()) return 0;
    if(node1.inputs.find("add")!=node1.inputs.end()) return 0;
    if(node2.inputs.find("add")!=node2.inputs.end()) return 0;

    if(node1.params.at("strides").get_iarray()!=node2.params.at("strides").get_iarray()) return 0;
    
    if(node1.params.at("kernel_shape").get_iarray()!=node2.params.at("kernel_shape").get_iarray()) return 0;
    if(node1.params.at("pads").get_iarray()!=node2.params.at("pads").get_iarray()) return 0;

    if(node1.get_output().shape.dims.at(2)!=node2.get_output().shape.dims.at(2)) return 0;
    if(node1.get_output().shape.dims.at(3)!=node2.get_output().shape.dims.at(3)) return 0;

    int neq_relu=0;
    if(node1.params.has_key("has_relu")!=node2.params.has_key("has_relu")) 
	{
		neq_relu=1;
		return 0;
	}

	{
			auto &shape1= node1.get_input("W").shape;
			auto &shape2= node2.get_input("W").shape;
			for(int i=1;i<=3;i++)
			{
					if(shape1.dims.at(i)!=shape2.dims.at(i)) return 0;
			}
	}

    if(neq_relu)
    {
        release_relu(graph,node_name1);
        release_relu(graph,node_name2);
    }


    auto &shape1= node1.get_input("W").shape;
    auto &shape2= node2.get_input("W").shape;

    auto &bshape1= node1.get_input("B").shape;
    auto &bshape2= node2.get_input("B").shape;
    assert(bshape1.dims.size()==1&&bshape2.dims.size()==1);



    auto &inits=graph.workspace->inits;

    string weight_tensor_name=graph.workspace->gen_var_name();
    Init & weight_init=graph.workspace->inits[weight_tensor_name];
    weight_init.name=weight_tensor_name;
    weight_init.shape=shape1;
    weight_init.shape.dims.at(0)+=shape2.dims.at(0);

    weight_init.lazy.has_value=1;
    string &w1_name=node1.inputs.at("W"),&w2_name=node2.inputs.at("W");
    weight_init.lazy.deps.push_back(w1_name);
    weight_init.lazy.deps.push_back(w2_name);
    weight_init.lazy.func=[w1_name,w2_name,&inits]()
    {
        return concat_vec( inits.at(w1_name).data,inits.at(w2_name).data);
    };
    //weight_init.data=concat_vec( inits.at(node1.inputs.at("W")).data,inits.at(node2.inputs.at("W")).data);

    string bias_tensor_name=graph.workspace->gen_var_name();
    Init & bias_init=graph.workspace->inits[bias_tensor_name];
    bias_init.name=bias_tensor_name;
    bias_init.shape=bshape1;
    bias_init.shape.dims.at(0)+=bshape2.dims.at(0);
    
    bias_init.lazy.has_value=1;
    string &b1_name=node1.inputs.at("B"),&b2_name=node2.inputs.at("B");
    bias_init.lazy.deps.push_back(b1_name);
    bias_init.lazy.deps.push_back(b2_name);
    bias_init.lazy.func=[b1_name,b2_name,&inits]()
    {
        return concat_vec( inits.at(b1_name).data,inits.at(b2_name).data);
    };
    //bias_init.data=concat_vec( inits.at(node1.inputs.at("B")).data,inits.at(node2.inputs.at("B")).data);


    string conv_name=graph.workspace->gen_node_name();
    assert(graph.tensors.find(conv_name)==graph.tensors.end());
    string split_name=graph.workspace->gen_node_name();
    assert(graph.tensors.find(split_name)==graph.tensors.end());
    Node &conv=graph.nodes[conv_name];
    Node &split=graph.nodes[split_name];

    conv=node1;
    conv.name=conv_name;
    conv.inputs["W"]=weight_tensor_name;
    conv.inputs["B"]=bias_tensor_name;
    assert(graph.tensors.find(conv_name+":0")==graph.tensors.end());
    conv.outputs["Y"]=conv_name+":0";

    split.type=SPLIT;
    split.name=split_name;
    split.inputs["X"]=conv.outputs["Y"];
    split.outputs["0"]=node1.outputs.at("Y");
    split.outputs["1"]=node2.outputs.at("Y");
    split.params["axis"].set_int(1);
    vector<int> sizes(2);
    vector<int> offsets(2);
    sizes[0]=shape1.dims.at(0);
    sizes[1]=shape2.dims.at(0);
    offsets[0]=0;
    offsets[1]=sizes[0];
    split.params["sizes"].set_iarray(sizes);
    split.params["offsets"].set_iarray(offsets);


    graph.nodes.erase(node1.name);
    graph.nodes.erase(node2.name);
    graph.post_process();

    //printf("merge success\n");

    //assert(graph.tensors.find(weight_tensor_name)==graph.tensors.end());
    //Tensor & weight_tensor=graph.tensors[weight_tensor_name];

    //assert(graph.tensors.find(bias_tensor_name)==graph.tensors.end());
    //Tensor & bias_tensor=graph.tensors[bias_tensor_name];







    return 1;

}

bool ConcatConcat::apply_inplace(Graph &graph, const string &name)
{
    ////printf("run1,%s\n",name.c_str());
    //vector<Graph > r;
    auto &node=graph.nodes.at(name);
    if(node.type!=CONCAT) return 0;
    if(node.get_output().read_by.size()!=1) return 0;
    auto &next_node=node.get_output().next_node();
    if(next_node.type!=CONCAT) return 0;

    ////printf("run2\n");

    int idx=stoi(node.get_output().read_by[0].idx);
    int size=node.input_size();
    assert(size>1);
    int shift=size-1;
    int next_size=next_node.input_size();
    auto &sizes_vec=next_node.params["sizes"].get_iarray();
    sizes_vec.resize(next_size+shift);
    auto &offsets_vec=next_node.params["offsets"].get_iarray();
    offsets_vec.resize(next_size+shift);

    for(int i=next_size-1;i>=idx+1;i--)
    {
        next_node.inputs[to_string(i+shift)]=next_node.inputs.at(to_string(i));
        sizes_vec[i+shift]=sizes_vec[i];
        offsets_vec[i+shift]=offsets_vec[i];
    }
    int last_offset;
    if(idx==0) last_offset=0;
    else last_offset=offsets_vec[idx];
    for(int i=0;i<size;i++)
    {
        next_node.inputs[to_string(i+idx)]=node.inputs.at(to_string(i));
        offsets_vec[i+idx]= node.params["offsets"].get_iarray()[i]+last_offset;
        sizes_vec[i+idx]=node.params["sizes"].get_iarray()[i];
    }
    graph.nodes.erase(name);
    graph.post_process();

    ////printf("run3\n");

    return 1;
}

bool SplitSplit::apply_inplace(Graph &graph, const string &name)
{
    ////printf("run1,%s\n",name.c_str());
    //vector<Graph > r;
    auto &node=graph.nodes.at(name);
    if(node.type!=SPLIT) return 0;
    if(node.get_input().read_by.size()!=1) return 0;
    if(node.get_input().write_by.idx=="") return 0;
    auto &last_node=graph.nodes.at(node.get_input().write_by.name);
    if(last_node.type!=SPLIT) return 0;

    ////printf("run2\n");

    int idx=stoi(node.get_input().write_by.idx);
    int size=node.output_size();
    assert(size>1);
    int shift=size-1;
    int last_size=last_node.output_size();
    auto &sizes_vec=last_node.params["sizes"].get_iarray();
    sizes_vec.resize(last_size+shift);
    auto &offsets_vec=last_node.params["offsets"].get_iarray();
    offsets_vec.resize(last_size+shift);

    for(int i=last_size-1;i>=idx+1;i--)
    {
        last_node.outputs[to_string(i+shift)]=last_node.outputs.at(to_string(i));
        sizes_vec[i+shift]=sizes_vec[i];
        offsets_vec[i+shift]=offsets_vec[i];
    }
    int last_offset;
    if(idx==0) last_offset=0;
    else last_offset=offsets_vec[idx];
    for(int i=0;i<size;i++)
    {
        last_node.outputs[to_string(i+idx)]=node.outputs.at(to_string(i));
        offsets_vec[i+idx]= node.params["offsets"].get_iarray()[i]+last_offset;
        sizes_vec[i+idx]=node.params["sizes"].get_iarray()[i];
    }
    graph.nodes.erase(name);
    graph.post_process();

    /////printf("run3\n");

    return 1;
}

/*
void greedy_fix_point(Graph & graph)
{
    ConvAdd s0;
    ConvRelu s1;
    SplitConcat s2;
    ConcatConcat s3;
    SplitSplit s4;

    int changed=1;
    while(changed)
    {
        changed=0;
        if(s0.apply_all_inplace(graph)) changed=1;
        if(s1.apply_all_inplace(graph)) changed=1;
        if(s2.apply_all_inplace(graph)) changed=1;
        if(s3.apply_all_inplace(graph)) changed=1;
        if(s4.apply_all_inplace(graph)) changed=1;
    }

    return ;
}*/

/*
void optimize(Graph & graph,const string &profiler_name,cost_func_t func)
{
    greedy_fix_point(graph);
    int changed=1;
    ConvConv ss1;
    ConvConvNeq ss2;
	int round=0;
    while(changed)
    {
		round++;
		printf("round %d\n",round);
        changed=0;
        deque<Graph> r;
        ss1.apply_all(graph,r);
        ss2.apply_all(graph,r);
		graph.memory_copy_optimize();
        double cost=get_cost(graph,profiler_name,func);
        int best=-1;
        for(int i=0;i<(int)r.size();i++)
        {
			//r[i].to_html("x.html");
			greedy_fix_point(r[i]);
  			r[i].memory_copy_optimize();
            double cost_new=get_cost(r[i],profiler_name,func);
            if(cost_new<cost)
            {
                cost=cost_new;
                best=i;
            }
        }
        if(best!=-1)
        {
            graph=r[best];
            changed=1;
        }
    }
    return ;
}
*/
