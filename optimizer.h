#pragma once

#include "common.h"
#include "graph.h"
#include "trans.h"
#include "profiler.h"

struct Optimizer;

struct Strategy
{
    Optimizer *optimizer=0;
    virtual void run(Graph &graph)=0;
};

struct StrategyManager
{
    unordered_map<string, unordered_map<string,Strategy*>> mp;
    static StrategyManager *p;

    StrategyManager();
    static StrategyManager * get_instance()
    {
        if(p==0)
        {
            p=new StrategyManager();
        }
        return p;
    }
    void register_strategy(string type, string name, Strategy * a)
    {
		//return ;
        assert(mp[type].find(name)==mp[type].end());
        mp[type][name]=a;
    }
    Strategy* get_strategy(string type,string name)
    {
		//return 0;
        return mp.at(type).at(name);
    }
};

struct Optimizer  //don't create more than one at the moment
{
    Strategy *trans=0;
    Strategy *mem=0;
    Strategy *select=0;

    Profiler *profiler;
    cost_func_t cost_func;

    Optimizer();
    void set_trans(string name)
    {
        trans=StrategyManager::get_instance()->get_strategy("trans",name);
        trans->optimizer=this;
    }
    void set_mem(string name)
    {
        mem=StrategyManager::get_instance()->get_strategy("mem",name);
        mem->optimizer=this;
    }
    void set_select(string name)
    {
        select=StrategyManager::get_instance()->get_strategy("select",name);
        select->optimizer=this;
    }
    void set_profiler(string name)
    {
        profiler=ProfilerManager::get_instance()->get_profiler(name);
    }
    void set_costfunc(cost_func_t func)
    {
		cost_func=func;
    }
    Graph optimize(Graph &graph)
    {
		//return graph;
        Graph result=graph;
        trans->run(result);
        result.resolve_lazy();
        return result;
    }
};
