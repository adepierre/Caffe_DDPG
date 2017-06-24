#pragma once

#include <vector>
#include <random>

#include <opencv2/core.hpp>

class Noise
{
public:
	
	Noise(const int &dim_, const float &mu_ = 0.0f, const float &theta_ = 0.15f, const float &sigma_ = 0.3f);

	~Noise();

	std::vector<float> GetNoise();

	void Reset();	

private:

	float dim;
	float mu;
	float theta;
	float sigma;

	std::vector<float> state;

	std::mt19937 random_gen;
	
};


