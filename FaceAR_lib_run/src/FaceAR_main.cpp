/////////////////////////////////////////////////////////////////////////////////

#include "MultiTrackCLM.h"


int main (int argc, char **argv)
{
    std::string mp4_file = "../data/test.mp4";
    std::vector<std::string> files;
    files.push_back(mp4_file);

    pre_main (files);

    return 0;
}

