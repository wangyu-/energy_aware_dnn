#pragma once

#include "graph.h"
#include "log.h"


void delete_node(Graph &graph,const string& tensor_name, const string &node_name);
bool merge_conv(Graph &graph, const string &node_name1,const string &node_name2);
bool merge_conv_neq(Graph &graph, const string &node_name1,const string &node_name2);
bool enlarge_conv(Graph &graph,const string &node_name1, const vector<int> &ker);

struct Trans
{
    string name;
    virtual int apply(Graph &graph, const string &name,deque <Graph >& r)
    {
        mylog(log_warn,"used undefined function apply()\n");
        short_pause();
        return 0;
    }
    virtual int apply_all(Graph &graph,deque <Graph >& r)
    {
        int cnt=0;
        for(auto &x:graph.nodes)
        {
            cnt+=apply(graph,x.first,r);
        }
        return cnt;
    }
    virtual bool apply_inplace(Graph &graph, const string &name)
    {
        mylog(log_warn,"used undefined function apply_inplace()\n");
        short_pause();
        return 0;
    }
    bool apply_all_inplace(Graph &graph)
    {
       // Graph g=graph;//no need to inplace here, it's not peformance bottole necks
        int changed=1;
        int cnt=0;
        while(changed)
        {
            //printf("!!!!!!!!!!!!!!!!!!!!!!1\n");
            //g.prt();
            changed=0;
            for(auto &x:graph.nodes)
            {
                int r=apply_inplace(graph,x.first);
                //assert(r.size()<=1);
                if(r)
                {
                    changed=1;
                    //cout<<"changed at node: "<<x.first<<endl;
                    cnt++;
                    break;
                }
            }
        }
        return cnt>0;
    }
};


struct ConvRelu:Trans
{
    ConvRelu()
    {
        name="ConvRelu";
    }
    virtual bool apply_inplace(Graph &graph, const string& name) override
    {
        //vector<Graph > r;
        //graph.fill_backward();
        auto &node=graph.nodes.at(name);
        if(node.type!=CONV) return 0;
        if(node.params.has_key("has_relu")) return 0;
        assert(node.output_size()==1);
        if(node.get_output().read_by.size()!=1) return 0;

        auto &next_node=node.get_output().next_node();
        auto &next_name=next_node.name;
        if(next_node.type!=RELU) return 0;

        //r.push_back(graph);
        //r[0]=graph;
        {
            //auto &graph=r[0];
            //graph.fill_backward();
            //auto &node=graph.nodes[name];
            //auto &next_node=graph.nodes[next_name];
            node.params["has_relu"].set_int(1);
            delete_node(graph,node.get_output().name,next_node.name);
            //delete_next_node(graph,node.name);
            graph.post_process();
        }
        return 1;
    }

};

struct ConvAdd:Trans
{
    ConvAdd()
    {
        name="ConvAdd";
    }
    virtual bool apply_inplace(Graph &graph, const string& name) override
    {
        //vector<Graph > r;
        //graph.fill_backward();
        auto &node=graph.nodes.at(name);
        if(node.type!=CONV) return 0;
        if(node.params.has_key("has_relu")) return 0;
        if(node.inputs.find("add")!=node.inputs.end()) return 0;
        assert(node.output_size()==1);
        if(node.get_output().read_by.size()!=1) return 0;

        auto &next_node=node.get_output().next_node();
        auto &next_name=next_node.name;
        if(next_node.type!=ADD) return 0;

        assert(next_node.input_size()==2);

        string my_idx=node.get_output().read_by[0].idx;
        string other_idx;
        assert(my_idx=="A"||my_idx=="B");
        if(my_idx=="A") other_idx="B";
        else other_idx="A"; 

        node.inputs["add"]=next_node.inputs.at(other_idx);
        
        delete_node(graph,node.get_output().name,next_node.name);
        graph.post_process();
        return 1;
    }

};

struct SplitConcat:Trans
{
    SplitConcat()
    {
        name="SplitConcat";
    }
    virtual bool apply_inplace(Graph &graph, const string &name) override
    {
        //vector<Graph > r;
        auto &node=graph.nodes.at(name);
        if(node.type!=SPLIT) return 0;
        
        {
            //Graph g=graph;
            //auto &node=g.nodes[name];
            int changed=0;
            int size=node.output_size();
            for(int i=0;i<size;i++)
            {
                if(node.get_output(to_string(i)).read_by.size()!=1) continue;
                //printf("1");
                auto & read1=node.get_output(to_string(i)).read_by.at(0);
                if(i+1>=node.output_size()) continue;
                //printf("2");
                if(node.get_output(to_string(i+1)).read_by.size()!=1) continue;
                //printf("3");
                auto & read2=node.get_output(to_string(i+1)).read_by.at(0);
                if(read1.name!=read2.name) continue;
                auto & next_node=graph.nodes.at(read1.name);
                if(next_node.type!=CONCAT) continue;
                //printf("4");
                int idx1=stoi(read1.idx),idx2=stoi(read2.idx);
                if(idx1+1!=idx2) continue;
				//printf("%d<%d,%d>\n",i,idx1,idx2);
                int size2=next_node.inputs.size();

                //r.push_back(graph);
                //auto &g=r[0];
                //auto & node=g.nodes[name];

                //printf("!!!!\n");
                //printf("<<%s>>",node.get_output(to_string(i+1)).name.c_str());
                //g.tensors.erase(node.get_output(to_string(i+1)).name);
                node.outputs.erase(to_string(i+1));
                for(int j=i+1;j+1<size;j++)
                {
                    node.outputs[to_string(j)]=node.outputs.at(to_string(j+1));
                    node.outputs.erase(to_string(j+1));

                }

                next_node.inputs.erase(to_string(idx1+1));
                for(int j=idx1+1;j+1<size2;j++)
                {
                    next_node.inputs[to_string(j)]=next_node.inputs.at(to_string(j+1));
                    next_node.inputs.erase(to_string(j+1));
                }

                {
						auto &vec1=node.params["sizes"].get_iarray();
						vec1[i]+=vec1[i+1];
						//printf("<%s>\n",node.name.c_str());
						vec1.erase(vec1.begin()+i+1);
						auto &vec2=node.params["offsets"].get_iarray();
						//printf("<%s>\n",node.name.c_str());
						vec2.erase(vec2.begin()+i+1);
                }
                {
						auto &vec1=next_node.params["sizes"].get_iarray();
						vec1[idx1]+=vec1[idx1+1];
						//printf("<%s>\n",node.name.c_str());
						vec1.erase(vec1.begin()+idx1+1);
						auto &vec2=next_node.params["offsets"].get_iarray();
						//printf("<%s>\n",node.name.c_str());
						vec2.erase(vec2.begin()+idx1+1);
                }
                graph.post_process();
                if(node.output_size()==1)
                {
                    delete_node(graph,node.get_input().name,node.name);
                    graph.post_process();
                }
                if(next_node.input_size()==1)
                {
                    delete_node(graph,next_node.get_input().name,next_node.name);
                    graph.post_process();
                }
                //r.push_back(g);
                return 1;
            }

        }

        return 0;
    }

};


struct ConcatConcat:Trans
{
    ConcatConcat()
    {
        name="ConcatConcat";
    }
    virtual bool apply_inplace(Graph &graph, const string &name) override;

};

struct SplitSplit:Trans
{
    SplitSplit()
    {
        name="SplitSplit";
    }
    virtual bool apply_inplace(Graph &graph, const string &name) override;

};

struct ConvConv:Trans
{
    ConvConv()
    {
        name="ConvConv";
    }
    virtual int apply(Graph &graph,const string& name, deque<Graph> &results) override
    {
        auto &node=graph.nodes.at(name);
        int original_size=results.size();
        if(node.type!=CONV) return 0;
        auto & read_by=node.get_input("X").read_by;
        if(read_by.size()==1) return 0;
        int has_sibling=0;
        for(int i=0;i<(int)read_by.size();i++)
        {
            if(read_by[i].name==name) continue;
            if(read_by[i].idx!="X") continue;
            if(graph.nodes[read_by[i].name].type==CONV) {has_sibling=1;break;}
        }
        if(!has_sibling) return 0;
        Graph candidate=graph;
        for(int i=0;i<(int)read_by.size();i++)
        {
            if(read_by[i].name==name) continue;
            if(read_by[i].idx!="X") continue;
			if(graph.nodes[read_by[i].name].type!=CONV) continue;
            int changed=merge_conv(candidate,name,read_by[i].name);
            if(changed)
            {
                results.push_back(candidate);
                candidate=graph;
            }
        }
        return (int)results.size()-original_size;
    }
};

struct ConvConvNeq:Trans
{
    ConvConvNeq()
    {
        name="ConvConvNeq";
    }
    virtual int apply(Graph &graph,const string& name, deque<Graph> &results) override
    {
        auto &node=graph.nodes.at(name);
        int original_size=results.size();
        if(node.type!=CONV) return 0;
        auto & read_by=node.get_input("X").read_by;
        if(read_by.size()==1) return 0;
        int has_sibling=0;
        for(int i=0;i<(int)read_by.size();i++)
        {
            if(read_by[i].name==name) continue;
            if(read_by[i].idx!="X") continue;
            if(graph.nodes[read_by[i].name].type==CONV) {has_sibling=1;break;}
        }
        if(!has_sibling) return 0;
        Graph candidate=graph;
        for(int i=0;i<(int)read_by.size();i++)
        {
            if(read_by[i].name==name) continue;
            if(read_by[i].idx!="X") continue;
			if(graph.nodes[read_by[i].name].type!=CONV) continue;
            int changed2=merge_conv_neq(candidate,name,read_by[i].name);
            if(changed2)
            {
                results.push_back(candidate);
                candidate=graph;
            }
        }
        return (int)results.size()-original_size;
    }
};

/*
void greedy_fix_point(Graph & graph);

void optimize(Graph & graph,const string &profiler_name,cost_func_t func);
*/