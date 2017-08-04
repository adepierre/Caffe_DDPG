#pragma once

#include <vector>
#include <random>

#include <opencv2/core.hpp>

struct Transition
{
	Transition(const std::vector<float> &state_, const float &reward_, bool terminal_)
	{
		state = state_;
		reward = reward_;
		terminal = terminal_;
	}

	std::vector<float> state;
	float reward;
	bool terminal;
};

class Continuous_Mountain_Car
{
public:
	
	Continuous_Mountain_Car(bool display, const int &FPS_ = 60);

	~Continuous_Mountain_Car();

	Transition Step(const std::vector<float> &action);

	std::vector<float> Reset();

	std::vector<float> GetState();

	inline int ActionDim() { return 1; }
	inline int StateDim() { return 2; }

private:
	float min_action;
	float max_action;
	float min_position;
	float max_position;
	float max_speed;
	float goal_position;
	float power;

	cv::Mat background_image;
	float pixel_width;
	float pixel_height;

	std::vector<float> state;

	std::mt19937 random_gen;
	bool display;
	int frame_speed;
};


