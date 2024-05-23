#include "nn.h"

#include <torch/script.h>
#include <torch/torch.h>

torch::jit::script::Module actor;
char actor_loaded = 0;
torch::jit::script::Module critic_1;
char critic_1_loaded = 0;
torch::jit::script::Module critic_2;
char critic_2_loaded = 0;

DLL__ void actor_load()
{
	actor = torch::jit::load("actor.pth");
	if (torch::cuda::is_available())
	{
		//actor = torch::jit::load("actor_cuda.pth");
		actor.to(torch::Device(torch::kCUDA));
		actor_loaded = actor_loaded | 2;
	}
	else
	{
		//actor = torch::jit::load("actor_cpu.pth");
		actor.to(torch::Device(torch::kCPU));
	}
	actor.eval();
	actor_loaded = actor_loaded | 1;
}

DLL__ void critic_1_load()
{
	critic_1 = torch::jit::load("critic_1.pth");
	if (torch::cuda::is_available())
	{
		//critic_1 = torch::jit::load("critic_1_cuda.pth");
		critic_1.to(torch::Device(torch::kCUDA));
		critic_1_loaded = critic_1_loaded | 2;
	}
	else
	{
		//critic_1 = torch::jit::load("critic_1_cpu.pth");
		critic_1.to(torch::Device(torch::kCPU));
	}
	critic_1.eval();
	critic_1_loaded = critic_1_loaded | 1;
}


DLL__ void critic_2_load()
{
	critic_2 = torch::jit::load("critic_2.pth");
	if (torch::cuda::is_available())
	{
		//critic_2 = torch::jit::load("critic_1_cuda.pth");
		critic_2.to(torch::Device(torch::kCUDA));
		critic_2_loaded = critic_1_loaded | 2;
	}
	else
	{
		//critic_2 = torch::jit::load("critic_1_cpu.pth");
		critic_2.to(torch::Device(torch::kCPU));
	}
	critic_2.eval();
	critic_2_loaded = critic_2_loaded | 1;
}

DLL__ void actor_step(std::vector<float>& input, std::vector<float>& output)
{
	if ((actor_loaded & 1) == 0)
	{
		actor_load();
	}
	at::Tensor input_ = at::from_blob(input.data(), { 1,810 });
	at::Tensor output_ = actor.forward({ input_.to((actor_loaded & 2) == 0 ? torch::Device(torch::kCPU) : torch::Device(torch::kCUDA)) }).toTensor().to(torch::Device(torch::kCPU));
	output = std::vector<float>(output_.data_ptr<float>(), output_.data_ptr<float>() + output_.numel());
}

DLL__ void critic_1_step(std::vector<float>& state, std::vector<float>& action, std::vector<float>& output)
{
	if ((critic_1_loaded & 1) == 0)
	{
		critic_1_load();
	}
	std::vector<float>input;
	input.reserve(1083);
	input.insert(input.end(), state.begin(), state.end());
	input.insert(input.end(), action.begin(), action.end());
	at::Tensor input_ = at::from_blob(input.data(), { 1,1082 });
	at::Tensor output_ = critic_1.forward({ input_.to((critic_1_loaded & 2) == 0 ? torch::Device(torch::kCPU) : torch::Device(torch::kCUDA)) }).toTensor().to(torch::Device(torch::kCPU));
	output = std::vector<float>(output_.data_ptr<float>(), output_.data_ptr<float>() + output_.numel());
}

DLL__ void critic_2_step(std::vector<float>& state, std::vector<float>& action, std::vector<float>& output)
{
	if ((critic_2_loaded & 1) == 0)
	{
		critic_2_load();
	}
	std::vector<float>input;
	input.reserve(1083);
	input.insert(input.end(), state.begin(), state.end());
	input.insert(input.end(), action.begin(), action.end());
	at::Tensor input_ = at::from_blob(input.data(), { 1,1082 });
	at::Tensor output_ = critic_2.forward({ input_.to((critic_2_loaded & 2) == 0 ? torch::Device(torch::kCPU) : torch::Device(torch::kCUDA)) }).toTensor().to(torch::Device(torch::kCPU));
	output = std::vector<float>(output_.data_ptr<float>(), output_.data_ptr<float>() + output_.numel());
}
