#pragma once

#include <bits/stdc++.h>
using namespace std;
#include <unistd.h>

#define DATATYPE float
#define CUDNN_DATATYPE CUDNN_DATA_FLOAT
const size_t CUDNN_DEFAULT_WORKSPACE_SIZE = (size_t)4 * 1024l * 1024l * 1024l;

//const int MAX_DIM=5;
struct Graph;
struct Var;
struct Node;
struct Workspace;
struct Weight;
struct Var;
struct Tensor;


struct Engine;
struct DeviceTensor;
struct ExecuterManager;



extern vector<DATATYPE> random_pool;;

const string CONV="Conv";
const string SPLIT="Split";
const string CONCAT="Concat";
const string RELU="Relu";
const string POOL="Pool";
const string ADD="Add";

inline void msleep(int a)
{
        usleep(a*1000);
}

inline void short_pause()
{
	//msleep(500);
}
inline long long get_current_time_us()
{
        timespec tmp_time;
        clock_gettime(CLOCK_MONOTONIC, &tmp_time);
        return tmp_time.tv_sec*1000*1000ll+tmp_time.tv_nsec/(1000l);
}
inline long long get_current_time_ms()
{
        timespec tmp_time;
        clock_gettime(CLOCK_MONOTONIC, &tmp_time);
        return tmp_time.tv_sec*1000ll+tmp_time.tv_nsec/(1000*1000l);
}


inline vector<DATATYPE> vec_from_file(string name)
{
	ifstream myfile(name);
	int size;
	myfile>>size;
	vector<DATATYPE> r;
	for(int i=0;i<size;i++)
	{
		DATATYPE tmp;
		myfile>>tmp;
		r.push_back(tmp);
	}
	return r;
}

inline void vec_to_file(vector<DATATYPE> vec,string name)
{
	ofstream myfile(name);
	myfile<<vec.size()<<endl;
	for(int i=0;i<(int)vec.size();i++)
	{
		myfile<<vec[i]<<endl;
	}
	return ;
}

inline vector<string> string_to_vec(string s,const char * sp) {
          vector<string> res;
          string str=s;
          char *p = strtok ((char *)str.c_str(),sp);
          while (p != NULL)
          {
                res.push_back(p);
                p = strtok(NULL, sp);
          }

          return res;
}

inline string vec_to_string(vector<int> &v) 
{

        stringstream ss;
        ss<<"(";
        for(int i=0;i<(int)v.size();i++)
        {
			if(i) ss<<",";
            ss<<v[i];
        }
        ss<<")";
        return ss.str();
}

struct Value
{
    int i;
    float f;
    string s;
    vector<int> ia;
    enum Type {NONE,INT,FLOAT,STRING,IARRAY};
    Type type=NONE;
    void clear()
    {
        type=NONE;
    }
    bool has_value()
    {
        return type!=NONE;
    }
    int get_int()
    {
        assert(type==INT);
        return i;
    }
    float get_float()
    {
        assert(type==FLOAT);
        return f;
    }
    string &get_string()
    {
        assert(type==STRING);
        return s;
    }
    vector<int> &get_iarray()
    {
        assert(type==IARRAY);
        return ia;
    }
    void set_int(int in)
    {
        type=INT;
        i=in;
    }
    void set_float(float in)
    {
        type=FLOAT;
        f=in;
    }
    void set_string(const string &in)
    {
        type=STRING;
        s=in;
    }
    void set_iarray(const vector<int> &in)
    {
        type=IARRAY;
        ia=in;
    }
    string to_string()
    {
        stringstream ss;
        if(type==NONE) ss<<"None";
        if(type==INT)  ss<<i;
        if(type==FLOAT) ss<<f;
        if(type==STRING) ss<<s;
        if(type==IARRAY)
        {
            ss<<"[";
            for(auto &a:ia)
                ss<<a<<",";
            ss<<"]";
        }
        return ss.str();
    }
};

struct NoCopy
{
	NoCopy()=default;
	NoCopy(const NoCopy&)=delete;
	void operator=(const NoCopy&)=delete;
};

struct measure_t
{
    bool invalid=0;
    double runtime=0.0;
    double energy=0.0;
    double power=0.0;
    void clear()
    {
        invalid=0;
        runtime=0.0;
        energy=0.0;
        power=0.0;
    }
    string to_string()
    {
        if(invalid)
        {
            return "invalid";
        }
        string r;
        r+="runtime="+::to_string(runtime);
        r+=",energy="+::to_string(energy);
        r+=",power="+::to_string(power);
        return r;
    }
    void from_string(string &s)
    {
        if(s=="invalid") 
        {
            invalid=1;
            runtime=0;power=0;energy=0;
        }
        else
        {
            auto vec=string_to_vec(s,",");
            assert(vec.size()==3);
            runtime=stod(string_to_vec(vec[0],"=")[1]);
            energy=stod(string_to_vec(vec[1],"=")[1]);
            power=stod(string_to_vec(vec[2],"=")[1]);
        }
    }
};

typedef function<double(const measure_t&)> cost_func_t;
typedef function<void(void)> void_func_t;

typedef function<vector<DATATYPE>(void)> lazy_func_t;
