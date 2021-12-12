#pragma once

#include "common.h"
#include "graph.h"
#include "log.h"
#include "pb/onnx.proto3.pb.h"

struct Helper
{
    string type;
    Helper()
    {
        type="Default";
    }
    virtual void shape_inference(Node &node)
    {
        assert(node.outputs.size()==1);
        assert(node.inputs.size()==1);
        auto in_name=node.inputs.begin()->second;
        auto out_name=node.outputs.begin()->second;
        node.graph->tensors[out_name].shape=node.graph->tensors[in_name].shape;
    }
    virtual void fill_node(onnx::NodeProto &pb_node,Node &node)
    {
        fill_inputs_outputs(pb_node,node);
        fill_attributes(pb_node,node);
        fill_attributes_post(pb_node,node);
    }
    
    virtual void fill_inputs_outputs(onnx::NodeProto &pb_node,Node &node)
    {
        for(int j=0;j<pb_node.input_size();j++)
        {
            node.inputs[to_string(j)]=pb_node.input(j);
        }
        for(int j=0;j<pb_node.output_size();j++)
        {
            node.outputs[to_string(j)]=pb_node.output(j);
        }
    }
    virtual void fill_attributes_post(onnx::NodeProto &pb_node,Node &node)
    {

    }
    virtual void fill_attributes(onnx::NodeProto &pb_node,Node &node)
    {
        int in_size = pb_node.input_size();
        int out_size = pb_node.output_size();
        for(int j=0; j<pb_node.attribute_size();j++)
        {
            auto &attr_name=pb_node.attribute(j).name();
            //cout<<pb_node.attribute(j).name()<<" ";
            //std::string json;
            //google::protobuf::util::MessageToJsonString(node, &json);
            //cout<<node.attribute(j).DebugString()<<" ";
            //cout<<node.attribute(j).type()<<" ";
            auto type=pb_node.attribute(j).type();
            if(type==::onnx::AttributeProto_AttributeType_INT)
            {
                node.params[attr_name].set_int(pb_node.attribute(j).i());
                //cout<<pb_node.attribute(j).i()<<" ";
            }
            else if(type==::onnx::AttributeProto_AttributeType_FLOAT)
            {
                node.params[attr_name].set_float(pb_node.attribute(j).f());
                //cout<<pb_node.attribute(j).i()<<" ";
            }
            else if(type==::onnx::AttributeProto_AttributeType_INTS)
            {
                vector<int> tmp;
                //cout<<"[";
                for(int k=0;k<pb_node.attribute(j).ints_size();k++)
                {
                    //cout<<pb_node.attribute(j).ints(k)<<" ";
                    tmp.push_back(pb_node.attribute(j).ints(k));
                }
                //cout<<"]";
                node.params[attr_name].set_iarray(tmp);
            }
            else
            {
                assert(0==1);
                //cout<<"unknow"<<endl;
            }
        }
    }
    virtual string get_key(Node &node)
    {
        assert(node.outputs.size()==1);
        string &output_name=node.outputs.begin()->second;
        auto &output_tensor=node.graph->tensors[output_name];

        return node.type+","+output_tensor.shape.to_string();
    }

};

struct HelperManager
{
    unordered_map<string,Helper*> mp;
    static HelperManager *p;
    HelperManager();
    static HelperManager * get_instance();
    void register_helper(Helper *helper);
    Helper *get_helper(string type);
};
