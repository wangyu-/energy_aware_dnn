#pragma once

#include "common.h"

#include "pb/onnx.proto3.pb.h"
#include "pb/model_file.proto3.pb.h"


struct Params:unordered_map<string,Value>
{
    bool has_key(const string &s)
    {
        return this->find(s)!=this->end();
    }
};
struct Node
{
    string type;
    string name;
    Params params;

    map<string,string> inputs;   // conv X->input101 W->weight102 B->bias103 etc
    map<string,string> outputs;  // Y->output104
    //vector<string> weights;

    int input_size()
    {
        return inputs.size();
    }
    int output_size()
    {
        return outputs.size();
    }

    Tensor& get_input(string idx);
    Tensor& get_output(string idx);
    Tensor& get_output();
    Tensor& get_input();

    //Tensor& get_weight(int idx);
    Graph * graph; //cached value

    string comment;
    struct extra_t
    {
        string algo_name="default";
        measure_t measure;
        map<string,measure_t> full_measure;
    }extra;

    string to_string()
    {
        stringstream ss;
        ss<<"node:"<<name<<"\t";
        ss<<"inputs: ";
        for(auto it=inputs.begin();it!=inputs.end();it++)
        {
            ss<<it->first<<"->"<<it->second;
            ss<<";";
        }
        ss<<"\t";
        ss<<"outputs: ";
        for(auto it=outputs.begin();it!=outputs.end();it++)
        {
            ss<<it->first<<"->"<<it->second;
            ss<<";";
        }
        ss<<"\t";
        ss<<"params: ";
        for(auto it=params.begin();it!=params.end();it++)
        {
            ss<<it->first <<"->"<<it->second.to_string()<<";";
        }
        ss<<endl;
        return ss.str();
    }
    void prt()
    {
        cout<<to_string();
    }

    string get_key();
};

struct Usage
{
    string name;
    string idx;
    string to_string()
    {
        stringstream ss;
        ss<<"("<<name<<","<<idx<<")";
        return ss.str();
    }
};

struct Shape
{
    vector<int> dims;
    static vector<int> to_strides(vector <int> &vec)
    {
        vector<int> strides(vec.size());
    	int mul=1;
		for(int i=(int)vec.size()-1;i>=0;i--)
		{
			strides[i]=mul;
			mul*=vec[i];
		}
        return strides;
    }
    vector<int> get_strides()
    {
        return to_strides(dims);
    }
    bool has_value()
    {
        return !dims.empty();
    }
    string to_string()
    {
        stringstream ss;
        ss<<"[";
        for(int i=0;i<(int)dims.size();i++)
        {
            if(i)ss<<",";
            ss<<dims[i];
        }
        ss<<"]";
        return ss.str();
    }
    string to_string2()
    {
        stringstream ss;
        //ss<<"[";
        for(int i=0;i<(int)dims.size();i++)
        {
            if(i)ss<<"x";
            ss<<dims[i];
        }
        //ss<<"]";
        return ss.str();
    }
    int size()
    {
        int r=1;
        assert(dims.size()!=0);
        for(int i=0;i<(int)dims.size();i++)
        {
            r*=dims[i];
        }
	return r;
    }
    int byte_size()
    {
        return size()*sizeof(DATATYPE);
    }
};

struct Tensor
{
    string name;
    Shape shape;
    Usage write_by;
    vector<Usage> read_by;
    Node &get_node();
    Node &next_node();
    bool is_constant();
    bool is_input();
    bool is_output();
    Graph * graph;//cached value

    int force_stride_scale=1;//for testing stride
    struct mem_t
    {
        string part_of;
        int offset=-1;
        int has_child=0;
    }mem;
    //int size=-1;

    string comment;
    string to_string()
    {
        stringstream ss;
        ss<<"Tensor: "<<name<<" ";
        ss<<"shape:"<<shape.to_string();
        ss<<"write_by: "<<write_by.to_string()<<" ";
        ss<<"read_by: [";
        for(auto &x:read_by)
        {
            ss<<x.to_string()<<" ";
        }
        ss<<"]"<<endl;
        return ss.str();
    }

};

struct Init
{
    string name;
    Shape shape;
    vector<DATATYPE> data;
    
    struct lazy_t
    {
        int has_value=0;
        lazy_func_t func;
        vector<string> deps;
    }lazy;


    string to_string()
    {
        stringstream ss; 
        ss<<"init"<<":"<<name<<" ";
        ss<<"shape:"<<shape.to_string();
        ss<<" ";
        for(int i=0;i<shape.size()&&i<10;i++)
            ss<<data[i]<<" ";
        ss<<endl;
        return ss.str();
    }
    void prt()
    {
        cout<<to_string();
    }
};

struct Graph
{
    map<string,Node> nodes;
    map<string,Tensor> tensors;
    map<string,Shape> inputs;
    map<string,Shape> outputs;
    Workspace *workspace;
    vector<string> topo_order;
    string comment;

    void clear_info()
    {
        for(auto &x:nodes)
        {
            x.second.extra.full_measure.clear();
            x.second.extra.measure.clear();
            x.second.comment.clear();
        }
        for(auto &x:tensors)
        {
            x.second.comment.clear();
        }
    }

    void clear()
    {
        nodes.clear();
        tensors.clear();
        inputs.clear();
        outputs.clear();
        topo_order.clear();
        comment.clear();
    }

    void resolve_lazy();
    void fill_backward()
    {
        for(auto &x:nodes)
            x.second.graph=this;
        for(auto &x:tensors)
            x.second.graph=this;
    }
    Graph();
    Graph(const Graph &b)
    {
        *this=b;
        //fill_backward();
    }
    Graph subgraph(vector<string> in,vector<string>out)
    {
        set<string> in_set,out_set;
        for(int i=0;i<(int)in.size();i++)
        {
            assert(tensors.find(in[i])!=tensors.end());
            in_set.insert(in[i]);
        }
        for(int i=0;i<(int)out.size();i++)
        {
            assert(tensors.find(out[i])!=tensors.end());
            out_set.insert(out[i]);
        }

        set<string> visited;
        
        std::function<void(string)> dfs= [&](string name)->void
        {
            if(visited.find(name)!=visited.end()) return;
            //topo_order.push_back(name);
            visited.insert(name);
            auto &node=nodes.at(name);
            for(auto &x:node.outputs)
            {
                auto &tensor=tensors[x.second];
                if(out_set.find(tensor.name)!=out_set.end())
                {
                    continue;
                }
                for(auto &y:tensor.read_by)
                {
                    dfs(y.name);
                }
            }
        };
        for(auto &x:in_set)
        {
            auto &tensor=tensors.at(x);
            for(auto &y:tensor.read_by)
            {
                dfs(y.name);
            }
        }
        Graph new_graph;
        new_graph.workspace=workspace;
        //printf("visit:");
        for(auto &x:visited)
        {
            new_graph.nodes[x]=nodes.at(x);
            //printf("<%s>",x.c_str());
        }
        for(auto &x:in_set)
            new_graph.inputs[x]=tensors.at(x).shape;
        for(auto &x:out_set)
            new_graph.outputs[x]=tensors.at(x).shape;
        new_graph.post_process();
        return new_graph;
        //printf("\n");
    }
    void  clean_copy_from(const Graph &b)
    {
        this->nodes=b.nodes;
        this->tensors=b.tensors;
        this->inputs=b.inputs;
        this->outputs=b.outputs;
        fill_backward();
    }
    Graph & operator= (const Graph &b)
    {
        clean_copy_from(b);
        this->workspace=b.workspace;
        this->topo_order=b.topo_order;
        return *this;
    }
    void post_process()
    {
        check_and_fill();
        topo_sort();
        shape_inference();
    }
    void post_process_preserve()
    {
        map<string,Node::extra_t> node_info;
        map<string,Tensor::mem_t> tensor_info;
        for(auto &x:nodes)
        {
            node_info[x.first]=x.second.extra;
        }
        for(auto &x:tensors)
        {
            tensor_info[x.first]=x.second.mem;
        }
        post_process();
        for(auto &x:node_info)
        {
            nodes.at(x.first).extra=x.second;
        }
        for(auto &x:tensor_info)
        {
            tensors.at(x.first).mem=x.second;
        }
    }
    void topo_sort()
    {
        topo_order.clear();
        unordered_map<string,int> pending_inputs;
        unordered_set<string> used;
        for(auto &x:nodes)
        {
            auto &node=x.second;
            pending_inputs[x.first]=node.inputs.size();
        }
        //cout<<"aaa"<<endl;
        std::function<void(string)> dfs= [&](string name)->void
        {
            if(used.find(name)!=used.end()) return;
            topo_order.push_back(name);
            used.insert(name);
            auto &node=nodes.at(name);
            for(auto &x:node.outputs)
            {
                auto &tensor=tensors[x.second];
                for(auto &y:tensor.read_by)
                {
                    pending_inputs[y.name]--;
                    assert(pending_inputs[y.name]>=0);
                    if(pending_inputs[y.name]==0) dfs(y.name);
                }
            }
        };
        //cout<<"bbb"<<endl;
        for(auto &x:tensors)
        {
            //cout<<"ccc"<<endl;
            auto &tensor=x.second;
            if(tensor.is_input()||tensor.is_constant())
            {
                for(auto &y:tensor.read_by)
                {
                   pending_inputs[y.name]--;
                   assert(pending_inputs[y.name]>=0);
                   if(pending_inputs[y.name]==0) dfs(y.name);
                }
            }
        }
        assert(topo_order.size()==nodes.size());

    }
    void shape_inference();
    void check_and_fill();
    void prt()
    {
        cout<<"----nodes----"<<endl;
        for(auto it=nodes.begin();it!=nodes.end();it++)
        {
            cout<<it->second.to_string();
        }
        cout<<"----inputs----"<<endl;
        for(auto it=inputs.begin();it!=inputs.end();it++)
        {
            cout<<"Inputs: "<<it->first<<"->"<<it->second.to_string()<<endl;
        }
        cout<<"----ouputs----"<<endl;
        for(auto it=outputs.begin();it!=outputs.end();it++)
        {
            cout<<"Outputs: "<<it->first<<"->"<<it->second.to_string()<<endl;
        }
        cout<<"----tensors----"<<endl;
        if(topo_order.size()==0)
        {
            for(auto &x:tensors)
            {
                cout<<x.second.to_string();
            }
        }
        else
        {
            for(auto &x:inputs)
            {
                cout<<tensors.at(x.first).to_string();
            }
            for(auto &x:topo_order)
            {
                for(auto &y: nodes.at(x).outputs)
                {
                    cout<<tensors.at(y.second).to_string();
                }
            }
        }
        cout<<"----topo_order----"<<endl;
        for(auto &x:topo_order)
        {
            cout<<x<<",";
        }
        cout<<endl;
    }
	void from_node(Node &node)   //graph from a single node for measuring
	{
			Graph &old_graph=*node.graph;
			this->workspace=old_graph.workspace;
			unordered_set<string> tensor_names;
			for(auto &x: node.inputs)
			{
					tensor_names.insert(x.second);
			}
			for(auto &x: node.outputs)
			{
					tensor_names.insert(x.second);
			}
			for(auto &x:tensor_names)
			{
					tensors[x]=old_graph.tensors[x];
					auto &tensor=tensors[x];
					tensor.graph=this;
					if(tensor.mem.part_of!="")
					{
							if(tensor_names.find(tensor.mem.part_of)==tensor_names.end())
							{
									tensor.mem.part_of="";
							}		
					}
			}
			nodes[node.name]=node;
			nodes[node.name].graph=this;
			topo_order.clear();
			topo_order.push_back(node.name);   //be super careful, this new graph doesn't contain whole info
	}	
    void from_onnx(string file_name);
    void save_model(string file_name);
    void load_model(string file_name);

    //void to_html(string file_name);
};

struct Workspace
{
    string magic;
    unordered_map<string,Init> inits;
    int counter_node=0;
    int counter_var=0;
    //int counter_weight=0;

    static Workspace *p;

    static Workspace *get_instance()
    {
        if(p==0)
        {
            p=new Workspace();
        }
        return p;
    }
    string gen_node_name()
    {
            counter_node++;
            string name=magic+"_Node_"+to_string(counter_node);
            return name;
    }
    string gen_var_name()
    {
            counter_var++;
            string name=magic+"_Var_"+to_string(counter_var);
            assert(inits.find(name)==inits.end());
            return name;
    }
    void prt()
    {
        for(auto it=inits.begin();it!=inits.end();it++)
        {
            it->second.prt();
        }
    }
    void resolve_lazy(const string &name)
    {
        auto &init=inits.at(name);
        if(!init.lazy.has_value)return ;
        for(int i=0;i<(int)init.lazy.deps.size();i++)
        {
            resolve_lazy(init.lazy.deps[i]);
        }
        init.data=init.lazy.func();
        init.lazy.has_value=0;
        return;
    }
    /*
    string gen_weight_name()
    {
            counter_weight++;
            string name=magic+"_Weight_"+to_string(counter_weight);
            return name;
    }*/
};

