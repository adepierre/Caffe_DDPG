#include <gflags/gflags.h>

#include <NN_Agent.h>
#include <Continuous_Montain_Car.h>
#include <Noise.h>

#include <deque>
#include <random>
#include <chrono>

//Common flags
DEFINE_int32(train, 1, "Whether we want to train the agent or test it");
DEFINE_int32(display, 0, "Whether we want to display the screen or not");
DEFINE_int32(FPS, 2000, "Max number of images displayed per second when display is on. 0 to activate frame by frame mode.");

//Training flags
DEFINE_string(solver_actor, "solver_actor.prototxt", "Caffe solver file for actor net");
DEFINE_string(solver_critic, "solver_critic.prototxt", "Caffe solver file for actor net");
DEFINE_int32(memory_size, 100000, "Maximum size of the replay buffer");
DEFINE_int32(noise_iter_decay, 75000, "Iteration used for noise intensity (intensity = (noise_iter_decay - current iter) / noise_iter_decay");
DEFINE_int32(batch_size, 24, "Number of samples in one training pass");
DEFINE_string(log_file, "log.csv", "File to log during training");
DEFINE_double(target_net_update_rate, 0.001, "Rate for soft update of the target net during training");
DEFINE_double(gamma, 0.95, "Reward discount factor");
DEFINE_int32(max_len_episode, 2000, "Maximum number of step in one episode (-1 to deactivate)");
DEFINE_int32(num_episodes, 500, "Number of training episodes");

//Testing flags
DEFINE_string(model_actor, "", "Caffe prototxt to define actor net architecture");
DEFINE_string(model_critic, "", "Caffe prototxt to define critic net architecture");
DEFINE_string(weights_actor, "", "Trained weights to load into the actor net (.caffemodel)");
DEFINE_string(weights_critic, "", "Trained weights to load into the critic net (.caffemodel)");
DEFINE_int32(test_episode, 25, "Number of episode to test on");

int main(int argc, char** argv)
{
	//Use that if you want to run on GPU, but for a simple net like this one,
	//CPU is fast enough
/*
#ifdef CPU_ONLY
	caffe::Caffe::set_mode(caffe::Caffe::CPU);
#else
	caffe::Caffe::set_mode(caffe::Caffe::GPU);
#endif
*/

	caffe::Caffe::set_mode(caffe::Caffe::CPU);
	gflags::ParseCommandLineFlags(&argc, &argv, true);
	google::InitGoogleLogging(argv[0]);
	
	//In test mode we don't need all the caffe stuff
	if (!FLAGS_train)
	{
		for (int i = 0; i < google::NUM_SEVERITIES; ++i)
		{
			google::SetLogDestination(i, "");
		}
	}
	else
	{
		google::LogToStderr();
	}

	Continuous_Mountain_Car *env;
	NN_Agent *agent;

	std::vector<std::vector<float> > state;
	std::vector<float> action;

	if (FLAGS_train)
	{
		std::ofstream log_file(FLAGS_log_file);

		if (log_file.is_open())
		{
			log_file << "Step;Episode;Episode length;Reward" << std::endl;
		}

		std::deque<std::vector<float> > replay_states;
		std::deque<float> replay_rewards;
		std::deque<std::vector<float> > replay_actions;
		std::deque<bool> replay_terminals;

		std::vector<std::vector<float> > batch_states(FLAGS_batch_size, std::vector<float>(2, 0.0f));
		std::vector<std::vector<float> > batch_actions(FLAGS_batch_size, std::vector<float>(1, 0.0f));
		std::vector<float> batch_rewards(FLAGS_batch_size, 0.0f);
		std::vector<bool> batch_terminals(FLAGS_batch_size, false);
		std::vector<std::vector<float> > batch_states2(FLAGS_batch_size, std::vector<float>(2, 0.0f));

		env = new Continuous_Mountain_Car(FLAGS_display, FLAGS_FPS);
		agent = new NN_Agent(FLAGS_solver_actor, FLAGS_solver_critic, FLAGS_target_net_update_rate, FLAGS_gamma);
		std::mt19937 random_gen = std::mt19937(std::chrono::high_resolution_clock::now().time_since_epoch().count());

		Noise noise(env->ActionDim());

		state = std::vector<std::vector<float> >(1, std::vector<float>(env->StateDim(), 0.0f));
		action = std::vector<float>(env->ActionDim(), 0.0f);

		int step = 0;
		int step_episode = 0;
		int episode = 0;
		float reward = 0.0f;

		while (episode < FLAGS_num_episodes)
		{
			//Playing phase
			state[0] = env->GetState();

			replay_states.push_back(state[0]);

			if (replay_states.size() > FLAGS_memory_size)
			{
				replay_states.pop_front();
				replay_rewards.pop_front();
				replay_actions.pop_front();
				replay_terminals.pop_front();
			}

			action = agent->PredictActor(state, false)[0];
			std::vector<float> current_noise = noise.GetNoise();

			for (size_t i = 0; i < action.size(); ++i)
			{
				action[i] += std::max(-1.0f, std::min(1.0f, current_noise[i] * std::max(0.2f, (FLAGS_noise_iter_decay - step) / (float)FLAGS_noise_iter_decay)));

				if (std::isnan(action[i]))
				{
					std::cerr << "Warning: Action " << i << " is NaN. Set it to 0 instead." << std::endl;
					action[i] = 0.0f;
				}
			}

			Transition transition = env->Step(action);

			reward += transition.reward;
			step++;
			step_episode++;

			if (step_episode == FLAGS_max_len_episode)
			{
				transition.terminal = true;
			}

			if (transition.terminal)
			{
				log_file << step << ";" << episode << ";" << step_episode << ";" << reward << std::endl;
				episode++;
				step_episode = 0;
				reward = 0;
				env->Reset();
			}

			replay_rewards.push_back(transition.reward);
			replay_actions.push_back(action);
			replay_terminals.push_back(transition.terminal);

			// Training phase
			if (replay_states.size() > FLAGS_batch_size)
			{
				for (int i = 0; i < FLAGS_batch_size; ++i)
				{
					int index = std::uniform_int_distribution<int>(0, replay_states.size() - 2)(random_gen);
					batch_states[i] = replay_states[index];
					batch_actions[i] = replay_actions[index];
					batch_rewards[i] = replay_rewards[index];
					batch_terminals[i] = replay_terminals[index];
					batch_states2[i] = replay_states[index + 1];
				}

				agent->Train(batch_states, batch_actions, batch_rewards, batch_terminals, batch_states2);
			}
		}
	}
	else
	{
		env = new Continuous_Mountain_Car(FLAGS_display, FLAGS_FPS);
		agent = new NN_Agent(FLAGS_model_actor, FLAGS_model_critic, FLAGS_weights_actor, FLAGS_weights_critic);

		state = std::vector<std::vector<float> >(1, std::vector<float>(env->StateDim(), 0.0f));
		action = std::vector<float>(env->ActionDim(), 0.0f);
	}

	float mean_reward = 0.0f;
	float mean_length = 0.0f;
	float max_reward = 0.0f;
	int min_length = FLAGS_max_len_episode;
	float episode_reward = 0.0f;
	int episode_length = 0;

	for (int i = 0; i < FLAGS_test_episode; ++i)
	{
		while (true)
		{
			//Playing phase
			state[0] = env->GetState();
			episode_length++;

			action = agent->PredictActor(state, false)[0];

			Transition transition = env->Step(action);

			episode_reward += transition.reward;

			if (episode_length == FLAGS_max_len_episode)
			{
				transition.terminal = true;
			}

			if (transition.terminal)
			{
				mean_reward += episode_reward;

				if (episode_reward > max_reward)
				{
					max_reward = episode_reward;
				}

				episode_reward = 0;

				mean_length += episode_length;

				if (episode_length < min_length)
				{
					min_length = episode_length;
				}

				episode_length = 0;

				env->Reset();
				break;
			}
		}
	}
	mean_reward /= FLAGS_test_episode;
	mean_length /= FLAGS_test_episode;

	std::cout << "Max reward: " << max_reward << std::endl;
	std::cout << "Min length: " << min_length << " steps" << std::endl << std::endl;

	std::cout << "Mean reward: " << mean_reward << std::endl;
	std::cout << "Mean length: " << mean_length << " steps" << std::endl;
	std::cout << "Press any key to leave..." << std::endl;
	char a;
	std::cin >> a;

	delete env;
	delete agent;
}