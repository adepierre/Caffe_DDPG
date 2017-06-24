#include <chrono>
#include "Noise.h"

Noise::Noise(const int & dim_, const float & mu_, const float & theta_, const float & sigma_):
	dim(dim_), mu(mu_), theta(theta_), sigma(sigma_)
{
	random_gen = std::mt19937(std::chrono::high_resolution_clock::now().time_since_epoch().count());

	Reset();
}

Noise::~Noise()
{
}

std::vector<float> Noise::GetNoise()
{
	for (size_t i = 0; i < state.size(); ++i)
	{
		state[i] += theta * (mu - state[i]) + sigma * std::normal_distribution<float>(0.0f, 1.0f)(random_gen);
	}

	return state;
}

void Noise::Reset()
{
	state = std::vector<float>(dim, mu);
}
