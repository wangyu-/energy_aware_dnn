#pragma once

#include "common.h"
#include "graph.h"

struct Visualizer
{
    bool compact_mode=0;
    bool show_measure=1;
    void set_compact_mode(bool is_enable)
    {
        compact_mode=is_enable;
    }
    void set_show_measure(bool is_enable)
    {
        show_measure=is_enable;
    }
    void visualize(Graph &graph, string file_name);
};
