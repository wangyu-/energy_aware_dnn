#include "optimizer.h"
#include "engine.h"
StrategyManager * StrategyManager::p=0;


struct EmptyStrategy:Strategy
{
    virtual void run(Graph &graph)
    {
    }
};

struct Memopt:Strategy
{
    virtual void run(Graph &graph)
    {
        auto &tensors=graph.tensors;
        auto &nodes=graph.nodes;
        auto &topo_order=graph.topo_order;

        const int simple_mode=0;
        //map<string,string> mp={{"Split","back"},{"Concat","forward"},{"Identity","forward"},{"Flattern","either"}};
        map<string,string> mp={{"Split","back"},{"Concat","forward"},{"Identity","forward"},{"Flattern","either"}};
        for (auto &x : tensors)
        {
            //assert(x.second.part_of.empty());
            x.second.mem.part_of.clear();
        }
        for(auto &x : topo_order)
        {
            auto &node=nodes.at(x);
            auto &type=node.type;
            if(mp.find(type)==mp.end()) continue;
            decltype(node.inputs) *dsts0;
            decltype(node.inputs) *srcs0;
            if(mp[type]=="back")
            {
                dsts0=&node.inputs;
                srcs0=&node.outputs;
            }
            else if(mp[type]=="forward")
            {
                dsts0=&node.outputs;
                srcs0=&node.inputs;
            }
            else
            { 
                continue;
            }
            auto dsts=*dsts0;
            auto srcs=*srcs0;

            int cnt=0;
            assert(dsts.size()==1);
            auto &dst=tensors.at(dsts.begin()->second);
            if(simple_mode&&dst.mem.has_child) continue;   //todo seems like this is not necessary
            for(auto &y: srcs)
            {
                int idx;
                if(srcs.size()==1) idx=0;
                else idx=stoi(y.first);
                auto &src=tensors.at(y.second);
                extern int protect_input_output;
                if(protect_input_output)
                {
                        if(src.is_input()||src.is_output()) continue; //is input or output of graph
                }
                if(src.is_constant()) continue;
                if(simple_mode && src.mem.has_child) continue;
                if(src.mem.part_of=="") 
                {
                    cnt++;
                    src.mem.part_of=dst.name;
                    if(node.params.has_key("offsets"))
                        src.mem.offset=node.params["offsets"].get_iarray()[idx];
                    else 
                    {
                        assert(srcs.size()==1);
                        src.mem.offset=0;
                    }
                    //src.size=node.params["sizes"].get_iarray()[idx];
                }
            }
            if(cnt>0&&simple_mode)
            {
                dst.mem.has_child=1;
                //dst.part_of="#root";
            }
        }
    }
};

struct LocalSelect:Strategy
{
    virtual void run(Graph &graph)
    {
        auto &profiler=optimizer->profiler;
        auto cost_func=optimizer->cost_func;
        for(auto &x:graph.nodes)
        {
            //cout<<"\nnode:"<<x.first<<endl;
            vector<string> impl_list=ExecuterManager::get_instance()->get_exec_name_list(x.second.type);
            double min=999999999999.0;
            for(int i=0;i<(int)impl_list.size();i++)
            {
                //cout<<impl_list[i]<<": ";
                auto r=profiler-> do_cached_measure_node(x.second,impl_list[i]);
                if(r.invalid) continue;
                //cout<<impl_list[i]<<r.to_string()<<endl;
                double cost=cost_func(r);
                if(cost<min)
                {
                    min=cost;
                    //impl_mp[x.first]=impl_list[i];
                    x.second.extra.algo_name=impl_list[i];
                    //x.second.extra.measure=r;
                }
            }
            //cout<<"best:"<<impl_mp[x.first]<<endl;
            //x.second.prt();
        }
    }
};

struct GlobalSelect:Strategy
{
    virtual void run(Graph &graph)
    {
        auto &profiler=optimizer->profiler;
        auto cost_func=optimizer->cost_func;
        
        auto best_measure=profiler->model(graph);

        int changed=1;
        while(changed)
        {
            changed=0;
            for(auto &x:graph.nodes)
            {

                auto &node=x.second;
                auto current_node_measure=profiler->do_cached_measure_node(node,node.extra.algo_name);
                vector<string> impl_list=ExecuterManager::get_instance()->get_exec_name_list(x.second.type);
                for(int i=0;i<(int)impl_list.size();i++)
                {

                    if(impl_list[i]==node.extra.algo_name) continue;
                    auto other_node_measure=profiler->do_cached_measure_node(node,impl_list[i]);
                    if(other_node_measure.invalid) continue;

                    auto new_measure=best_measure;
                    new_measure.energy-=current_node_measure.energy;
                    new_measure.energy+=other_node_measure.energy;

                    new_measure.runtime-=current_node_measure.runtime;
                    new_measure.runtime+=other_node_measure.runtime;

                    new_measure.power=new_measure.energy/new_measure.runtime;

                    if(cost_func(new_measure)<cost_func(best_measure))
                    {
                        best_measure=new_measure;
						current_node_measure=other_node_measure;
                        node.extra.algo_name=impl_list[i];
                        changed=1;
                    }
                }
            }

        }
    }
};

struct NoTrans:Strategy
{
    virtual void run(Graph &graph)
    {
        optimizer->mem->run(graph);
        optimizer->select->run(graph);
    }
};

struct RuleOnly:Strategy
{
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
    }
    virtual void run(Graph &graph)
    {
        greedy_fix_point(graph);
        optimizer->mem->run(graph);
        optimizer->select->run(graph);
    }
};

struct FullTrans:RuleOnly
{
    void inner(Graph &graph)
    {
        greedy_fix_point(graph);
        optimizer->mem->run(graph);
        optimizer->select->run(graph);
    }
    virtual void run(Graph & graph)
    {
        auto & profiler=optimizer->profiler;
        auto & cost_func=optimizer->cost_func;
        inner(graph);
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

            double cost= cost_func( profiler->model(graph));
            int best=-1;
            for(int i=0;i<(int)r.size();i++)
            {
                //r[i].to_html("x.html");
                inner(r[i]);
                double cost_new= cost_func( profiler->model(r[i]));
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
};


Optimizer::Optimizer()
{
    set_trans("full");
    set_mem("default");
    set_select("global");
    set_profiler("Advanced");
    cost_func=[&](const measure_t& m){return m.runtime;};
}

StrategyManager::StrategyManager()
{
    register_strategy("trans","none",new NoTrans);
    register_strategy("trans","rule",new RuleOnly);
    register_strategy("trans","full",new FullTrans);

    register_strategy("mem","none",new EmptyStrategy);
    register_strategy("mem","default",new Memopt);
    
    register_strategy("select","none",new EmptyStrategy);
    register_strategy("select","local",new LocalSelect);
    register_strategy("select","global",new GlobalSelect);
}
