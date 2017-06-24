#include "Continuous_Montain_Car.h"

#include <chrono>
#include <algorithm>

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>


Continuous_Mountain_Car::Continuous_Mountain_Car(bool display_, const int &FPS_) : display(display_)
{
	frame_speed = FPS_ == 0 ? 0 : std::max(1, (int)(1000.0 / FPS_));

	min_action = -1.0f;
	max_action = 1.0f;
	min_position = -1.2f;
	max_position = 0.6f;
	max_speed = 0.07;
	goal_position = 0.45f;
	power = 0.0015f;

	state = std::vector<float>(2, 0.0f);

	random_gen = std::mt19937(std::chrono::high_resolution_clock::now().time_since_epoch().count());

	if (display)
	{
		background_image = cv::Mat::zeros(cv::Size(600, 400), CV_8UC3);

		pixel_width = (max_position - min_position) / background_image.cols;
		pixel_height = 2.5f / background_image.rows;

		for (int i = 0; i < background_image.cols; ++i)
		{
			float value = sin(3 * (min_position + i * pixel_width));
			int first_pixel = (int)((1.25f - value) / pixel_height);
			for (int j = first_pixel; j < background_image.rows; ++j)
			{
				background_image.at<cv::Vec3b>(j, i)[1] = 255;
			}
		}

		//Speed circle
		cv::circle(background_image, cv::Point(65, 350), 40, cv::Scalar(0, 0, 0), 5);
		cv::line(background_image, cv::Point(65, 350 - 40), cv::Point(65, 350 - 30), cv::Scalar(0, 0, 0), 5);
		cv::line(background_image, cv::Point(65 - 40.0, 350), cv::Point(65 - 30.0, 350), cv::Scalar(0, 0, 0), 5);
		cv::line(background_image, cv::Point(65 + 40.0, 350), cv::Point(65 + 30.0, 350), cv::Scalar(0, 0, 0), 5);

		//Acceleration
		cv::rectangle(background_image, cv::Rect(375, 325, 200, 50), cv::Scalar(0, 0, 0), 5);
		cv::line(background_image, cv::Point(475, 325), cv::Point(475, 375), cv::Scalar(0, 0, 0), 2);

		//Goal marker
		cv::rectangle(background_image, cv::Rect((goal_position - min_position) / pixel_width, (1.25f - sin(3 * goal_position)) / pixel_height - 25, 5, 25), cv::Scalar(255, 255, 255), -1);
	}

	Reset();
}

Continuous_Mountain_Car::~Continuous_Mountain_Car()
{
}

Transition Continuous_Mountain_Car::Step(const std::vector<float> &action)
{
	float force = std::min(std::max(action[0], -1.0f), 1.0f);

	if (display)
	{
		cv::Mat current_image;
		background_image.copyTo(current_image);

		cv::circle(current_image, cv::Point((state[0] - min_position) / pixel_width, (1.25f - sin(3 * state[0])) / pixel_height - 5), 5, cv::Scalar(255, 0, 0), -1);

		double speed_angle = (state[1] - 1.0) / 2.0 * 3.14159265359;

		cv::line(current_image, cv::Point(65, 350), cv::Point(65 + 35 * cos(speed_angle), 350 + 35 * sin(speed_angle)), cv::Scalar(0, 0, 255), 2);
		if (force > 0)
		{
			cv::rectangle(current_image, cv::Rect(475, 329, force * 96, 42), cv::Scalar(0, 0, 255), -1);
		}
		else
		{
			cv::rectangle(current_image, cv::Rect(475 + force * 96, 329, -force * 96, 42), cv::Scalar(0, 0, 255), -1);
		}
		cv::imshow("Environment", current_image);
		cv::waitKey(frame_speed);
	}

	//Speed is stored between -1 and 1
	state[1] += (force * power - 0.0025 * cos(3 * state[0])) / max_speed;
	if (state[1] > 1.0f)
	{
		state[1] = 1.0f;
	}

	if (state[1] < -1.0f)
	{
		state[1] = -1.0f;
	}

	state[0] += state[1] * max_speed;

	if (state[0] > max_position)
	{
		state[0] = max_position;
	}

	if (state[0] < min_position)
	{
		state[0] = min_position;
	}

	if (state[0] == min_position && state[1] < 0)
	{
		state[1] = 0.0f;
	}

	bool done = state[0] >= goal_position;

	float reward = done ? 100.0f : -action[0] * action[0] * 0.1f;

	return Transition(state, reward, done);;
}

std::vector<float> Continuous_Mountain_Car::Reset()
{
	state[0] = std::uniform_real_distribution<float>(-0.6f, -0.4f)(random_gen);
	state[1] = 0.0f;

	return state;
}

std::vector<float> Continuous_Mountain_Car::GetState()
{
	return state;
}
