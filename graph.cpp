#include "graph.h"
#include "log.h"
#include "helper.h"

Workspace * Workspace::p=0;

Tensor &Node::get_input(string idx)
{
    return graph->tensors[inputs.at(idx)];
}

Tensor &Node::get_output(string idx)
{
    return graph->tensors[outputs.at(idx)];
}

Tensor &Node::get_input()
{
    return graph->tensors[inputs.begin()->second];
}

Tensor &Node::get_output()
{
    return graph->tensors[outputs.begin()->second];
}

string Node::get_key()
{
    return HelperManager::get_instance()->get_helper(type)->get_key(*this);
}
/*
Tensor& Node::get_weight(int idx)
{
    return graph->tensors[weights[idx]];
}*/

Node &Tensor::get_node()
{
    return graph->nodes[write_by.name];
}
Node &Tensor::next_node()
{
    assert(read_by.size()==1);
    return graph->nodes[read_by[0].name];
}

bool Tensor::is_constant()
{
    auto &inits = graph->workspace->inits;
    return inits.find(name) != inits.end();
}

bool Tensor::is_input()
{
    return graph->inputs.find(name) != graph->inputs.end();
}

bool Tensor::is_output()
{
    return graph->outputs.find(name) != graph->outputs.end();
}


void Graph::check_and_fill()
{
    tensors.clear();
    for (auto &x : inputs)
    {
        auto tensor_name = x.first;
        assert(tensors.find(tensor_name) == tensors.end());
        auto &tensor = tensors[tensor_name];
        tensor.name = tensor_name;
        tensor.write_by.name = "<input>";
        tensor.shape = x.second;
    }
    for (auto &x : nodes)
    {
        auto &node_name = x.first;
        auto &node = x.second;
        node.graph = this;

        for (auto &y : node.outputs)
        {
            auto &tensor_idx = y.first;
            auto &tensor_name = y.second;
            assert(tensors.find(tensor_name) == tensors.end());
            //cout<<"!!"<<tensor_name<<endl;
            assert(workspace->inits.find(tensor_name) == workspace->inits.end());
            auto &tensor = tensors[tensor_name];
            tensor.name = tensor_name;
            tensor.graph = node.graph;
            tensor.write_by.name = node_name;
            tensor.write_by.idx = tensor_idx;
        }
    }
    for (auto &x : nodes)
    {
        auto &node_name = x.first;
        auto &node = x.second;
        //printf("<%s>\n",x.first.c_str());
        for (auto &y : node.inputs)
        {
            auto &tensor_idx = y.first;
            auto &tensor_name = y.second;
            //cout<<"<"<<tensor_name<<">"<<endl;
            if (tensors.find(tensor_name) == tensors.end())
            {
                assert(workspace->inits.find(tensor_name) != workspace->inits.end());
                tensors[tensor_name].name = tensor_name;
                tensors[tensor_name].write_by.name = "<init>";
                tensors[tensor_name].shape = workspace->inits[tensor_name].shape;
            }
            else
            {
                assert(workspace->inits.find(tensor_name) == workspace->inits.end()); //graph input or node output shouldn't be overrid by inits
            }
            auto &tensor = tensors[tensor_name];
            tensor.graph = node.graph;
            tensor.read_by.emplace_back();
            tensor.read_by.back().name = node_name;
            tensor.read_by.back().idx = tensor_idx;
        }
    }

    for (auto &x : outputs)
    {
        auto tensor_name = x.first;
        assert(tensors.find(tensor_name) != tensors.end());
    }
}

Graph::Graph()
{
    workspace=Workspace::get_instance();
}
void Graph::resolve_lazy()
{
    for(auto &x:tensors)
    {
        if(workspace->inits.find(x.first)!=workspace->inits.end())
        {
            workspace->resolve_lazy(x.first);
        }
    }
}

