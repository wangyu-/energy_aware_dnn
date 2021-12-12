#pragma once

#include "common.h"
#include "graph.h"

extern int NAIVE_WARM_UP_TIMES;//used in naive measure
extern int NAIVE_MEASURE_TIMES;

extern int WAIT_BEFORE_POWER_MEASURE;//used in power thread
extern int READ_POWER_PERIOD;  //used in power thread

extern int check_per_runs;
extern int idle_time;
extern int stress_time;
extern int measure_time;// including WAIT_BEFORE_POWER_MEASURE

void start_check_power();

void stop_check_power();
double finish_check_power();


measure_t do_measure(Engine &engine);

void init_power_thread();
//void init_measure();

struct Profiler
{
    string name;
    virtual measure_t do_measure(void_func_t func)=0;
    measure_t do_cached_measure_node(Node node,string impl_name);
    measure_t do_cached_measure(string key,void_func_t func);
    measure_t model(Graph& graph)
    {
            measure_t r;
            //SelectorManager::get_instance()->get_selector("Local")->select_impl(graph,profiler_name,func);
            for(auto &x:graph.nodes)
            {
                    assert(x.second.extra.algo_name!="");
                    measure_t tmp=do_cached_measure_node(x.second,x.second.extra.algo_name);
                    r.runtime+=tmp.runtime;
                    r.energy+=tmp.energy;
            }
			r.power=r.energy/r.runtime;
            return r;
    }
    void fill_measure_info(Graph& graph);
    void fill_measure_info_full(Graph& graph);
};

struct SimpleProfiler:Profiler
{
    SimpleProfiler()
    {
        name="Simple";
    }
    measure_t do_measure(void_func_t func);   
};

struct AdvancedProfiler:Profiler
{
    AdvancedProfiler()
    {
        name="Advanced";
    }
    measure_t do_measure(void_func_t func);   
};

struct ProfilerManager
{
    unordered_map<string,Profiler*> mp;
    static ProfilerManager *p;
    ProfilerManager()
    {
        register_profiler(new SimpleProfiler);
        register_profiler(new AdvancedProfiler);
    }
    static ProfilerManager * get_instance()
    {
        if(p==0) p=new ProfilerManager;
        return p;
    }
    void register_profiler(Profiler *profiler)
    {
        assert(mp.find(profiler->name)==mp.end());
        mp[profiler->name]=profiler;
    }
    Profiler *get_profiler(string name)
    {
        assert(mp.find(name)!=mp.end());
        return mp[name];
    }
};

/*
measure_t do_cached_measure(Node &node,string profiler_name,string impl_name);
void fill_measure_info(Graph& graph,string profiler_name);
void fill_measure_info_full(Graph& graph,string profiler_name);
measure_t get_total(Graph& graph);
double get_cost(Graph& graph,const string &profiler_name,cost_func_t func);
*/
