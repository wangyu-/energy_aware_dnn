#include "visualizer.h"

string wrap(string a)
{
    stringstream ss;
    ss << "\"" << a << "\"";
    return ss.str();
}
string map_to_json(map<string, string> mp)
{
    stringstream ss;
    ss << "{";
    for (auto it = mp.begin(); it != mp.end(); it++)
    {
        if (it != mp.begin())
            ss << ",";
        ss << it->first << ":" << wrap(it->second);
    }
    ss << "}";
    return ss.str();
}
string map_to_json2(map<string, string> mp)
{
    stringstream ss;
    ss << "{";
    for (auto it = mp.begin(); it != mp.end(); it++)
    {
        if (it != mp.begin())
            ss << ",";
        ss << it->first << ":" << wrap(it->second);
    }
	ss<<",curve: d3.curveBasis";
    ss << "}";
    return ss.str();
}
string gen_setnode(string name, map<string, string> mp)
{
    stringstream ss;
    ss << "g.setNode(" << wrap(name) << "," << map_to_json(mp) << ");" << endl;
    //sprintf(buf,"g.setNode(\"%s\",  { label: \"%s\\n%s\", shape: \"rect\",labelStyle: \"font-size: 1.5em\" })\n",
    return ss.str();
}
string gen_setedge(string from, string to, map<string, string> mp)
{
    stringstream ss;
    ss << "g.setEdge(" << wrap(from) << "," << wrap(to) << "," << map_to_json(mp) << ");" << endl;
    //sprintf(buf,"g.setNode(\"%s\",  { label: \"%s\\n%s\", shape: \"rect\",labelStyle: \"font-size: 1.5em\" })\n",
    return ss.str();
}
string gen_setedge2(string from, string to, map<string, string> mp,string id)
{
    stringstream ss;
    ss << "g.setEdge(" << wrap(from) << "," << wrap(to) << "," << map_to_json(mp) << ",\""<<id<<"\");" << endl;
    //sprintf(buf,"g.setNode(\"%s\",  { label: \"%s\\n%s\", shape: \"rect\",labelStyle: \"font-size: 1.5em\" })\n",
    return ss.str();
}
string multi_line(vector<string> vec)
{
    stringstream ss;
    for (int i = 0; i < (int)vec.size(); i++)
    {
        if (i)
            ss << "\\n";
        ss << vec[i];
    }
    return ss.str();
}
string get_color(Node &node)
{
    string type = node.type;
    map<string, string> mp = {{"Split", "#AED6F1"}, {"Concat", "#F9E79F"}, {"Conv", "#DAF7A6"}, {"Pool", "#E8DAEF"}};
    if (mp.find(type) == mp.end())
    {
        return "#FFFFFF";
    }
    else
        return mp[type];
}

string get_color(Tensor &tensor)
{
    //return "#EAEDED";
    //if(tensor.name=="12"||tensor.name=="7") return "#D98880";
    if(tensor.mem.has_child) return "#D98880";
    if(tensor.mem.part_of=="") return "#EAEDED";
    else return "#F2D7D5"; 
}

void Visualizer::visualize(Graph &graph, string file_name)
{
    //extern int compact;
    //const int compact=0;
    ifstream head("template/head.txt");
    string head_str((std::istreambuf_iterator<char>(head)),
                    std::istreambuf_iterator<char>());
    ifstream tail("template/tail.txt");
    string tail_str((std::istreambuf_iterator<char>(tail)),
                    std::istreambuf_iterator<char>());
    //cout<<head_str<<endl;
    //cout<<tail_str<<endl;

    ofstream out(file_name);
    out << head_str;

    string SP = "\\n";

	map<string,int> topo_idx;
    auto &topo_order=graph.topo_order;
	if(!topo_order.empty())
	{
		for(int i=0;i<(int)topo_order.size();i++)
		{
			topo_idx[topo_order[i]]=i;
		}
	}
    auto &nodes=graph.nodes;
    auto &workspace=graph.workspace;
    auto &tensors=graph.tensors;
    auto &inputs=graph.inputs;
    auto &outputs=graph.outputs;
    for (auto &x : nodes)
    {
        auto &node=x.second;
        string name = node.name;
        string type = node.type;
        string description;
        int cnt = 0;
        for (auto &y : node.inputs)
        {
            if (workspace->inits.find(y.second) == workspace->inits.end())
                continue;
            cnt++;
        }
		if(!topo_idx.empty())
		{
			description+="topo_idx: "+to_string(topo_idx[name])+"<br>";
		}
		if(node.extra.algo_name!="")
		{
			description+="algo_name: "+node.extra.algo_name+"<br>";
		}
        if (cnt > 0)
            description += "-----weights:-----<br>";
        for (auto &y : node.inputs)
        {
            if (workspace->inits.find(y.second) == workspace->inits.end())
                continue;
            //sprintf(buf,"g.setEdge(\"%s\", \"%s\",  { label: \"%s\"})\n", y.second.c_str(),x.second.name.c_str(),  y.first.c_str());
            description += y.first.c_str();
            description += ":  ";
            description += tensors.at(y.second).shape.to_string();
            description += "  ";
            description += y.second.c_str();
            description += "<br>";
        }
        if (node.params.size() > 0)
            description += "-----attributes-----<br>";
        for (auto &y : node.params)
        {
            description += y.first;
            description += ":  ";
            description += y.second.to_string();
            description += "<br>";
        }
        string label;
        label+=type;
        if(type=="Pool")
        {
            string subtype=node.params.at("subtype").get_string();
            label=subtype;
        }
        if(type=="Conv"||type=="Pool")
        {
            auto v=node.params.at("kernel_shape").get_iarray();
            string ker=" ";
			if(type=="Conv")
			{
				auto &weight_shape=node.get_input("W").shape;
				ker+=to_string(weight_shape.dims.at(0));
				ker+="x";
			}
            if(type=="Pool")
            {
                //label+=SP;
                //label+="("+node.params.at("subtype").get_string()+")";
            }
			ker+=to_string(v.at(0));
            ker+="x";
            ker+=to_string(v.at(1));
			label+=ker;
            //label+=SP+"kernel: "+ker;
        }
        if(type=="Conv")
        {
            if(node.params.has_key("has_relu"))
			{
					label+=SP;
					label+="Relu";
			}
        }
		if(show_measure&&node.extra.measure.runtime!=0)
		{
			label+=SP;
			label+="---------------";
			label+=SP;
			char tmp[100];
			sprintf(tmp,"time: %.4fms ",node.extra.measure.runtime);
			label+=tmp;
			label+=SP;
			//label+="energy:";
			sprintf(tmp,"energy: %.4fJ",node.extra.measure.energy);
			label+=tmp;	
			//label+=to_string(node.extra.runtime);
		}
        if(node.extra.full_measure.size()!=0)
		{
            description += "-----measures-----<br>";
            char tmp[100];
            for(auto &x:node.extra.full_measure)
            {
                sprintf(tmp,"%s: %.4fms %.4fJ<br>",x.first.c_str(),x.second.runtime,x.second.energy);
            	description+=tmp;
            }
		}

        //label+=SP;
        //label+="&nbsp&nbsp&nbsp&nbsp";
        map<string, string> mp = {
            //{"labelType", "html"},
            {"label", label},
            {"shape", "rect"},
            //{"labelStyle","font-size: 1.2em;"},
            {"description", description + node.comment},
            {"rx", "10"},
            {"ry", "10"},
            //{"height","50.5"},
            //{"width","110.5"},
            {"style", "fill:" + get_color(node)}};
        out << gen_setnode(name, mp);
    }

    //out<<"g.nodes().forEach(function(v) {var node = g.node(v);node.rx = node.ry = 10;});"<<endl;

    for (auto &x : tensors)
    {
        auto &tensor=x.second;
        if (workspace->inits.find(tensor.name) != workspace->inits.end())
            continue;
        string name = tensor.name;
        if(compact_mode)
        {
            if(inputs.find(name)==inputs.end() &&outputs.find(name)==outputs.end()) continue;
        }
        string description=tensor.shape.to_string() + "<br>" ;
        if(tensor.mem.part_of!="")
        {
            description+="part_of: "+tensor.mem.part_of+"<br>";
            description+="offset: "+to_string(tensor.mem.offset)+"<br>";
            //description+="size: "+to_string(x.second.size)+"<br>";
        }
        description+=tensor.comment;
        map<string, string> mp = {
            {"label", " "+tensor.shape.to_string()},
            {"description", description},
            {"labelStyle", "font-size: 0.7em;"},
            //{"label",x.second.shape.to_string()},
            {"shape", "ellipse"},
            {"style", "fill:" + get_color(tensor)},
            {"height","30.5"},
            //{"width","80.5"},

        };
        out << gen_setnode(name, mp);
    }
	int counter=0;
    if(compact_mode)
    {
        for (auto &x : nodes)
        {
            auto &node=x.second;
            string from=node.name;

            for (auto &y : node.outputs)
            {
                if (workspace->inits.find(y.second) != workspace->inits.end())
                    continue;
                //sprintf(buf,"g.setEdge(\"%s\", \"%s\",  { label: \"%s\"})\n", x.second.name.c_str(),y.second.c_str(),  y.first.c_str());
                auto &tensor=tensors.at(y.second);
                for(auto &z:tensor.read_by)
                {
                    string to=z.name;
                    //out << gen_setedge(from, to, {});
                    counter++;
                    out << gen_setedge2(from, to, {{"label", y.first+"->"+z.idx}},"edge"+to_string(counter));
                    //out << gen_setedge(from, to, {{"label", tensor.shape.to_string2()}});
                }
            }
        }
    }
    if(1)
    {
        for (auto &x : nodes)
        {
            for (auto &y : x.second.inputs)
            {
                if (workspace->inits.find(y.second) != workspace->inits.end())
                    continue;
                if(compact_mode && inputs.find(y.second)==inputs.end()) continue;
                //sprintf(buf,"g.setEdge(\"%s\", \"%s\",  { label: \"%s\"})\n", y.second.c_str(),x.second.name.c_str(),  y.first.c_str());
                out << gen_setedge(y.second, x.second.name, {{"label", y.first}});
            }

            for (auto &y : x.second.outputs)
            {
                if (workspace->inits.find(y.second) != workspace->inits.end())
                    continue;
                if(compact_mode && outputs.find(y.second)==outputs.end()) continue;
                //sprintf(buf,"g.setEdge(\"%s\", \"%s\",  { label: \"%s\"})\n", x.second.name.c_str(),y.second.c_str(),  y.first.c_str());
                out << gen_setedge(x.second.name, y.second, {{"label", y.first}});
            }
        }
    }

    out << tail_str;
    out.close();
}