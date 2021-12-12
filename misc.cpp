#include "common.h"
#include "profiler.h"
#include "engine.h"

vector<DATATYPE> random_pool0;
vector<DATATYPE> random_pool;

void init()
{
    std::random_device rd;
    std::default_random_engine eng(rd());
    std::uniform_real_distribution<> distr(-0.5, 0.5);
  

  const int rsize=20*1024*1024;
  const int scale=10;
  random_pool0.resize(rsize);
  random_pool.resize(scale*rsize);
  for(int i=0;i<(int)random_pool0.size();i++)
  {
    //random_pool[i]=0;
    random_pool0[i]=distr(eng);   //be aware it's pool0 here
    //random_pool0[i]=(-1.0+2.0*(rand()%32767)/32767.0)/10.0;
    //random_pool0[i]=(rand()%32767)/32767.0;
    //random_pool0[i]=(rand()%1000)/1000.0;
    //if(i%2==0) random_pool0[i]=0;
  }
  for(int i=0;i<scale;i++)
  {
    memcpy(&random_pool[i*rsize],random_pool0.data(), rsize*sizeof(DATATYPE));
  }

  init_power_thread();
  cudnnWorkspace::get_instance();
  Workspace::get_instance();

}
