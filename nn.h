#pragma once

#include <vector>

#ifdef GPNN_EXPORTS
#define DLL__ __declspec(dllexport)
#else
#define DLL__ __declspec(dllimport)
#endif


DLL__ void actor_load();

DLL__ void critic_1_load();

DLL__ void critic_2_load();

DLL__ void actor_step(std::vector<float>& input, std::vector<float>& output);

DLL__ void critic_1_step(std::vector<float>& state, std::vector<float>& action, std::vector<float>& output);

DLL__ void critic_2_step(std::vector<float>& state, std::vector<float>& action, std::vector<float>& output);