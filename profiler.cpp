#include "common.h"
#include "engine.h"
#include "cuda_common.h"
#include "profiler.h"

ProfilerManager * ProfilerManager::p=0;

std::atomic<int> check_power_stop(1);
pthread_mutex_t power_mutex = PTHREAD_MUTEX_INITIALIZER;
pthread_t power_pid;
vector<double> power_vec;

int NAIVE_WARM_UP_TIMES=10;//used in naive measure
int NAIVE_MEASURE_TIMES=100;

int WAIT_BEFORE_POWER_MEASURE=100;//used in power thread
int READ_POWER_PERIOD=10;  //used in power thread

int check_per_runs=50;
int idle_time=2000;
int stress_time=500;
int measure_time=WAIT_BEFORE_POWER_MEASURE+1000;// including WAIT_BEFORE_POWER_MEASURE

void *check_power(void *)
{
        while(1)
        {
                while(check_power_stop==1)
                {
                        usleep(2*1000);
                }
                pthread_mutex_lock(&power_mutex);

                msleep(WAIT_BEFORE_POWER_MEASURE);//time before measure

                while(1)
                {
                        double power=-1;
                        FILE *fp;
                        fp = fopen("/tmp/yw7/power_result","r");
                        assert(fp!=0);
                        assert(fscanf(fp,"%lf",&power)==1);
                        assert(power!=-1);
                        if(check_power_stop==1) break;
                        power_vec.push_back(power);
                        pclose(fp);
                        msleep(READ_POWER_PERIOD);
                }
                pthread_mutex_unlock(&power_mutex);
        }
}
void before_check_power()
{
        pthread_mutex_lock(&power_mutex);
}
void start_check_power()
{
        check_power_stop=0;
        pthread_mutex_unlock(&power_mutex);
}

void stop_check_power()
{
        check_power_stop=1;
}
double finish_check_power()
{ 
        check_power_stop=1;
        pthread_mutex_lock(&power_mutex);
        double sum=0,avg;
        assert(power_vec.size()>1);
        printf("[");
        //for(int i=0;i<(int)power_vec.size()-1;i++)
        int new_size= (int)power_vec.size() -max(1, int(power_vec.size()*0.05) );
        for(int i=0;i<new_size;i++)
        {
                sum+=power_vec[i];
                printf("%.2f, ",power_vec[i]);
        }
        printf("]\n");
        avg=sum/new_size;
        power_vec.clear();
        return avg;
}

void init_power_thread()
{
   before_check_power();
   int ret = pthread_create(&power_pid,NULL,check_power,NULL);
   if(ret != 0){
     printf("create pthread error\n");
     exit(1);
   }

}

struct cudaEventWrapper:NoCopy
{
    cudaEvent_t event;
    cudaEventWrapper()
    {
        checkCUDA(cudaEventCreate(&event));
    }
    ~cudaEventWrapper()
    {
        checkCUDA(cudaEventDestroy(event));
    }
};



measure_t SimpleProfiler::do_measure(void_func_t func)
{
  measure_t r;
  cudaEventWrapper startEvent0,endEvent0;
  auto &startEvent=startEvent0.event;
  auto &endEvent=endEvent0.event;
  
  msleep(50);

  checkCUDA(cudaDeviceSynchronize());
  for (int i = 0; i < NAIVE_WARM_UP_TIMES; i++) {
      func();
  }
  
  checkCUDA(cudaDeviceSynchronize());
  checkCUDA(cudaEventRecord(startEvent));
  for (int i = 0; i < NAIVE_MEASURE_TIMES; i++) {
      func();
      /////checkCUDA(cudaDeviceSynchronize());    //todo: only for whole graph measure
  }

  checkCUDA(cudaEventRecord(endEvent));
  checkCUDA(cudaEventSynchronize(endEvent));
  float milliseconds;
  cudaEventElapsedTime(&milliseconds, startEvent, endEvent);
  r.runtime=milliseconds/NAIVE_MEASURE_TIMES;
  //printf("navive_time=%f\n",r.runtime);
  return r;
}


measure_t AdvancedProfiler::do_measure(void_func_t func)
{

        measure_t r;
        cudaEventWrapper startEvent0,endEvent0;
        auto &startEvent=startEvent0.event;
        auto &endEvent=endEvent0.event;

        msleep(idle_time);

        double current_time;
        current_time=get_current_time_ms();
        for (int i = 0; ; i++) {
                if(i%check_per_runs==0&&get_current_time_ms()-current_time>stress_time) break;
                func();
        }
        checkCUDA(cudaDeviceSynchronize());
  int times=0;
  //printf("begin\n");
  //checkCUDA(cudaDeviceSynchronize());
  checkCUDA(cudaEventRecord(startEvent));
  current_time=get_current_time_ms();
  start_check_power();
  for (times = 0; ; times++) {
    if(times%check_per_runs==0&&(get_current_time_ms())-current_time>measure_time) break;
    //engine.inference();
    func();
    //////checkCUDA(cudaDeviceSynchronize());    //todo: only for whole graph measure
  }
  stop_check_power();
  checkCUDA(cudaEventRecord(endEvent));
  checkCUDA(cudaEventSynchronize(endEvent));
  float gpu_time;
  cudaEventElapsedTime(&gpu_time, startEvent, endEvent);
  //printf("end\n");
  double power=finish_check_power();
  double runtime=gpu_time/times;
  double energy=power*runtime;
  //printf("runtime=%f, power=%f, energy=%f\n",runtime,power,energy);
  r.power=power;
  r.runtime=runtime;
  r.energy=energy;

  return r;
}

struct DB
{
    unordered_map<string,measure_t> mp;
    static DB *p;
    DB()
    {
        
        ifstream db_input;
        db_input.open("db.txt");
        string line;
        vector<string> db_vec;

        int cnt=0;
        while (std::getline(db_input, line))
        {
                //printf("%s\n",line.c_str());
                db_vec.push_back(line);
                cnt++;
        }
        printf("%d lines read from db.\n",cnt);
        db_input.close();
        for(int i=0;i<(int)db_vec.size();i++)
        {
                auto vec=string_to_vec(db_vec[i],"|");
                assert(vec.size()==2); 
                string key=vec[0];
                measure_t r;
                r.from_string(vec[1]);
                assert(!has(key));
                mp[key]=r;  //don't use insert here  
        }

    }
    static DB * get_instance()
    {
        if(p==0) p=new DB;
        return p;
    }
    bool has(string &key)
    {
        if(mp.find(key)==mp.end()) return false;
        else return true;
    }
    measure_t get(string &key)
    {
            assert(has(key));
            return mp.at(key);
    }
    void insert(string &key,measure_t &r)
    {
            assert(!has(key));
            mp[key]=r;
    		ofstream db_output;
        	db_output.open("db.txt", std::ios_base::app);
            db_output<<key<<"|"<<r.to_string()<<endl;
            db_output.flush();
			db_output.close();
    }
};
DB * DB::p=0;

/*
measure_t do_cached_measure(Node &node,string profiler_name,string impl_name)
{
        Profiler* profiler=ProfilerManager::get_instance()->get_profiler(profiler_name);
        string key=node.get_key();
        key+="-"+impl_name;
        key+="-"+profiler_name;
        DB*db=DB::get_instance();
        if(db->has(key))
        {
                return db->get(key);
        }
        else
        {
				const int n=2;
                Engine engine[n];
				for(int i=0;i<n;i++) engine[i].from_node(node,impl_name);
                measure_t r;
                void_func_t func=[&](){
				for(int i=0;i<n;i++) engine[i].inference();
				};
                if(engine[0].check_valid()==false)
                {
                        r.invalid=1;
                }
                else
                {
                        r=profiler->do_measure(func);
						r.runtime/=n;
						r.energy/=n;
						cout<<"measured: "<<node.name<<"    key: "<<key<<"    "<<r.to_string()<<endl;
                }
                db->insert(key,r);
                return r;
        }

}*/


void Profiler::fill_measure_info(Graph& graph)
{
        for(auto &x:graph.nodes)
        {
                //auto tmp=x.second.extra.measure;
                x.second.extra.measure=do_cached_measure_node(x.second,x.second.extra.algo_name);
                //x.second.extra.measure=tmp;
                //r.runtime+=tmp.runtime;
                //r.energy+=tmp.energy;
        }
}
void Profiler::fill_measure_info_full(Graph& graph)
{
        for(auto &x:graph.nodes)
        {
                x.second.extra.measure=do_cached_measure_node(x.second,x.second.extra.algo_name);
                vector<string> impl_list=ExecuterManager::get_instance()->get_exec_name_list(x.second.type);
                for(int i=0;i<(int)impl_list.size();i++)
                {
                        auto r=do_cached_measure_node(x.second,impl_list[i]);
                        if(r.invalid) continue;
                        x.second.extra.full_measure[impl_list[i]]=r;
                }
        }
}

/*
measure_t get_total(Graph& graph)
{
        measure_t r;        
        for(auto &x:graph.nodes)
        {
                auto tmp=x.second.extra.measure;
                //measure_t tmp=do_cached_measure(x.second,profiler_name,x.second.extra.algo_name);
                //x.second.extra.measure=tmp;
                r.runtime+=tmp.runtime;
                r.energy+=tmp.energy;
        }
        return r;
}

double get_cost(Graph& graph,const string& profiler_name,cost_func_t func)
{
        measure_t r;
        SelectorManager::get_instance()->get_selector("Local")->select_impl(graph,profiler_name,func);
        for(auto &x:graph.nodes)
        {
                measure_t tmp=do_cached_measure(x.second,profiler_name,x.second.extra.algo_name);
                r.runtime+=tmp.runtime;
                r.energy+=tmp.energy;
        }
        return func(r);
}*/

set<string> measure_set;
measure_t Profiler::do_cached_measure_node(Node node,string impl_name)
{
        string key=node.get_key();
        key+="-"+impl_name;
        key+="-"+name;
		measure_set.insert(key);
        DB*db=DB::get_instance();
        if(db->has(key))
        {
                return db->get(key);
        }
        else
        {
                const int n=2;
                Engine engine[n];
                for(int i=0;i<n;i++) engine[i].from_node(node,impl_name);
                measure_t r;
                void_func_t func=[&](){
                for(int i=0;i<n;i++) engine[i].inference();
                };
                if(engine[0].check_valid()==false)
                {
                        r.invalid=1;
                }
                else
                {
                        r=do_measure(func);
                        r.runtime/=n;
                        r.energy/=n;
                        cout<<"measured: "<<node.name<<"    key: "<<key<<"    "<<r.to_string()<<endl;
                }
                db->insert(key,r);
                return r;
        }
}

measure_t Profiler::do_cached_measure(string key,void_func_t func)
{
        DB*db=DB::get_instance();
        if(db->has(key))
        {
                return db->get(key);
        }
        else
        {
                measure_t r;
                r=do_measure(func);
                db->insert(key,r);
                return r;
        }
}
