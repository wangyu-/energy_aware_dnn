#pragma once

#include "common.h"
#include "log.h"
#include "graph.h"
#include "cuda_common.h"

struct Executer
{
    string name;
    Engine *engine;
    virtual bool check_valid()
    {
        return true;
    }
    virtual void inference()=0;
    virtual Executer* clone()=0;
    virtual void init(Node &n)=0;
    virtual ~Executer(){};
};


struct DeviceTensor:NoCopy
{
	DeviceTensor()=default;
	DeviceTensor(const DeviceTensor&)=delete;
	void operator=(const DeviceTensor&)=delete;
    ~DeviceTensor()
    {
        if(data_ptr!=0)
        {
            cudaFree(data_ptr);
        }
    }
    string name;
    Shape shape;
    vector<int> strides_override;
    vector<DATATYPE> data;
    Engine *engine=0;
	vector<int> get_strides_or_override()
	{
		if(strides_override.size()!=0) return strides_override;
		return shape.get_strides();
	}
    void prt()
    {
	    cout<<shape.to_string();
	    if(data.size()!=0)
	    cout<<data[0]<<" "<<data[1];
	    cout<<endl;
    }
    //Engine * engine;
    int force_stride_scale=1; //force stride for batch dim
    void *data_ptr=0;

    void alloc_on_device()//device memeroy support stride
    {
        //assert(stride_scale==1);
        assert(data_ptr==0);
        checkCUDA(cudaMalloc(&data_ptr, shape.byte_size()*force_stride_scale));
    }
    void alloc_on_host()
    {
        assert(force_stride_scale==1); //host memory doesn't support stride, we can relax this in furture
        assert(data.size()==0);
	    data.resize(shape.size());
    }
	void fill_with_random()
	{
		cudaMemcpy(data_ptr, random_pool.data(), shape.byte_size(), cudaMemcpyHostToDevice);
	}
    void from_tensor(Tensor & tensor);
    void host_to_device()
    {
        assert(force_stride_scale==1); //we can implemet strides copy in furture
        assert(data_ptr!=0);
	    if((int)data.size()!=shape.size())  {mylog(log_fatal,"data.size()!=shape.size()\n");exit(-1);}
        cudaMemcpy(data_ptr,data.data(),shape.byte_size(),cudaMemcpyHostToDevice);
    }
    void host_to_device_async()
    {
        assert(force_stride_scale==1); //we can implemet strides copy in furture
        assert(data_ptr!=0);
	    if((int)data.size()!=shape.size())  {mylog(log_fatal,"data.size()!=shape.size()\n");exit(-1);}
        cudaMemcpyAsync(data_ptr,data.data(),shape.byte_size(),cudaMemcpyHostToDevice);
    }
    void host_to_device_async(DATATYPE * data)
    {
        assert(force_stride_scale==1); //we can implemet strides copy in furture
        assert(data_ptr!=0);
	    //if((int)data.size()!=shape.size())  {mylog(log_fatal,"data.size()!=shape.size()\n");exit(-1);}
        cudaMemcpyAsync(data_ptr,data,shape.byte_size(),cudaMemcpyHostToDevice);
    }
    void device_to_host()
    {
        assert(force_stride_scale==1); //we can implemet strides copy in furture
        assert(data_ptr!=0);
        assert((int)data.size()==shape.size());
        cudaMemcpy(data.data(),data_ptr,shape.byte_size(),cudaMemcpyDeviceToHost);
    }
    void device_to_host_async()
    {
        assert(force_stride_scale==1); //we can implemet strides copy in furture
        assert(data_ptr!=0);
        assert((int)data.size()==shape.size());
        cudaMemcpyAsync(data.data(),data_ptr,shape.byte_size(),cudaMemcpyDeviceToHost);
    }
    void device_to_host_async(DATATYPE * data)
    {
        assert(force_stride_scale==1); //we can implemet strides copy in furture
        assert(data_ptr!=0);
        //assert((int)data.size()==shape.size());
        cudaMemcpyAsync(data,data_ptr,shape.byte_size(),cudaMemcpyDeviceToHost);
    }
};

struct cudnnWorkspace
{
    void *ptr=0; //data_ptr
    size_t size=0;

    static cudnnWorkspace *p;
    static cudnnWorkspace* get_instance()
    {
        if(p==0)
        {
            p=new cudnnWorkspace();
            p->size=CUDNN_DEFAULT_WORKSPACE_SIZE;
            checkCUDA(cudaMalloc(&p->ptr, p->size));
        }
        return p;
    }
};

struct Engine
{
    unordered_map<string,DeviceTensor> tensors;
    vector<Executer*> executers;
    unordered_map<string,Executer*> executers_mp;
    cudnnHandle_t cudnn_handle;
    cublasHandle_t cublas_handle;
	cudaStream_t stream;
    vector<string> inputs;
    vector<string> outputs;

    int get_input_num()
    {
        return (int)inputs.size();
    }
    int get_output_num()
    {
        return (int)outputs.size();
    }
    DeviceTensor & get_input(int idx)
    {
        assert(idx<get_input_num());
        return tensors.at(inputs[idx]);
    }
    DeviceTensor & get_output(int idx)
    {
        assert(idx<get_output_num());
        return tensors.at(outputs[idx]);
    }

    cudnnWorkspace *cudnn_workspace;
    Engine()
    {
        cudnn_workspace=cudnnWorkspace::get_instance();
        //checkCUDA(cudaMalloc(&cudnn_workspace, CUDNN_WORKSPACE_SIZE));
    	checkCUDA(cudaStreamCreate(&stream));
        checkCUDNN(cudnnCreate(&cudnn_handle));
        checkCUDA(cublasCreate(&cublas_handle));
		//cublasSetStream(cublas_handle, stream);
		//cudnnSetStream(cudnn_handle, stream);
    }
    ~Engine()
    {
        for(int i=0;i<(int)executers.size();i++)
        {
            delete executers[i];
        }
        checkCUDNN(cudnnDestroy(cudnn_handle));
        checkCUDA(cublasDestroy(cublas_handle));
        checkCUDA(cudaStreamDestroy(stream));
    }

    void from_graph(Graph &graph);
    void from_node(Node &node,const string &impl_name);
    void inference()
    {
        for(int i=0;i<(int)executers.size();i++)
        {
	    //cout<<"executer: "<<executers[i]->name<<endl;
            executers[i]->inference();
    	    //checkCUDA(cudaDeviceSynchronize());
        }
    }
    bool check_valid()
    {
        for(int i=0;i<(int)executers.size();i++)
        {
            if(executers[i]->check_valid()==false) return false;
        }
        return true;
    }
};



struct ExecuterManager
{
    unordered_map<string, unordered_map<string,Executer*>> mp;
    static ExecuterManager *p;

    ExecuterManager();
    static ExecuterManager * get_instance()
    {
        if(p==0)
        {
            p=new ExecuterManager();
        }
        return p;
    }
    void register_exec(string type, string name, Executer * a)
    {
        assert(mp[type].find(name)==mp[type].end());
        mp[type][name]=a;
    }
    vector<string> get_exec_name_list(string type)
    {
        vector<string> impl_list;
        for(auto &x: mp.at(type))
        {
            impl_list.push_back(x.first);
        }
        return impl_list;
    }
    Executer*& get_exec(string type,string name)
    {
        return mp[type].at(name);
    }
};
