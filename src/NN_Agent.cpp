#include "NN_Agent.h"

#include <chrono>
#include <algorithm>
#include <numeric>
#include <random>

NN_Agent::NN_Agent(const std::string &solver_,
				   const float &tau_,
				   const float &gamma_):
			update_rate(tau_),
			gamma(gamma_)
{
	//Create caffe objects (solver + net)
	caffe::SolverParameter solver_param;
	caffe::ReadProtoFromTextFileOrDie(solver_, &solver_param);

	solver.reset(caffe::SolverRegistry<float>::CreateSolver(solver_param));

	net = solver->net();

	caffe::NetParameter net_param;

	net->ToProto(&net_param);

	net_param.mutable_state()->set_phase(caffe::Phase::TEST);

	target_net.reset(new caffe::Net<float>(net_param));
	
	target_net->CopyTrainedLayersFrom(net_param);

	//Save input and output blobs to find them easily later
	blob_state = net->blob_by_name("state");
	blob_action = net->blob_by_name("action");
	blob_q = net->blob_by_name("q_values");
	blob_label_q = net->blob_by_name("q_label");

	target_blob_state = target_net->blob_by_name("state");
	target_blob_action = target_net->blob_by_name("action");
	target_blob_q = target_net->blob_by_name("q_values");

	//Find the index of the last layer of the actor part of the net
	last_layer_actor = -1;
	for (size_t i = 0; i < net->layer_names().size(); i++)
	{
		if (net->layer_names()[i].compare("ActionLayer") == 0)
		{
			last_layer_actor = i;
			break;
		}
	}
	
	if (last_layer_actor == -1)
	{
		std::cerr << "Warning: No layer with name \"ActionLayer\" found. Make sure the layer which has the action blob as output is named ActionLayer." << std::endl;
	}
}

NN_Agent::NN_Agent(const std::string &model_file, 
				   const std::string &trained_file)
{
	net.reset(new caffe::Net<float>(model_file, caffe::TEST));
	
	if (!trained_file.empty())
	{
		net->CopyTrainedLayersFrom(trained_file);
	}
	
	blob_state = net->blob_by_name("state");
	blob_action = net->blob_by_name("action");
	blob_q = net->blob_by_name("q_values");

	//Find the index of the last layer of the actor part of the net
	last_layer_actor = -1;
	for (size_t i = 0; i < net->layer_names().size(); i++)
	{
		if (net->layer_names()[i].compare("ActionLayer") == 0)
		{
			last_layer_actor = i;
			break;
		}
	}

	if (last_layer_actor == -1)
	{
		std::cerr << "Warning: No layer with name \"ActionLayer\" found. Make sure the layer which has the action blob as output is named ActionLayer." << std::endl;
	}
}

NN_Agent::~NN_Agent()
{
	if (solver)
	{
		solver->Snapshot();
	}
}

void NN_Agent::Train(const std::vector<std::vector<float>> &states, const std::vector<std::vector<float>> &actions, const std::vector<float> &rewards, const std::vector<bool> &terminals, const std::vector<std::vector<float>> &states_after)
{
	std::vector<float> target_q = PredictCritic(states_after, PredictActor(states_after, true), true);

	std::vector<float> y(rewards.size(), 0.0f);

	for (size_t i = 0; i < rewards.size(); ++i)
	{
		if (terminals[i])
		{
			y[i] = rewards[i];
		}
		else
		{
			y[i] = rewards[i] + gamma * target_q[i];
		}
	}

	TrainCritic(states, actions, y);

	std::vector<float> grads = GetCriticGradient(states, PredictActor(states, false));

	//We negate the gradients because we want to perform gradient ascent 
	//(We want to follow the gradient that maximizes the Q value)
	for (size_t i = 0; i < grads.size(); ++i)
	{
		grads[i] = -grads[i];
	}
	
	TrainActor(states, grads);

	solver->iter_++;

	if (solver->iter() % 100 == 0)
	{
		float norm = 0;
		int param_number = 0;
		for (size_t i = 0; i < net->learnable_params().size(); i++)
		{
			for (size_t j = 0; j < net->learnable_params()[i]->count(); j++)
			{
				float data = net->learnable_params()[i]->cpu_data()[0];
				norm += data * data;
				param_number++;
			}
		}

		norm = sqrt(norm) / param_number;
		std::cout << "Mean L2 norm of the trainable layers of the net: " << norm << std::endl;
	}

	UpdateTargetNets();
}

std::vector<std::vector<float> > NN_Agent::PredictActor(const std::vector<std::vector<float> > &state, bool target)
{
	//A vector of batch size vectors of action dim size
	std::vector<std::vector<float> > prediction(target ? target_blob_action->num() : blob_action->num(), std::vector<float>(target ? target_blob_action->count() / target_blob_action->num() : blob_action->count() / blob_action->num(), 0.0f));

	float* mutable_cpu_data = target ? target_blob_state->mutable_cpu_data() : blob_state->mutable_cpu_data();
	int index = 0;

	for (size_t i = 0; i < state.size(); ++i)
	{
		for (size_t j = 0; j < state[i].size(); ++j)
		{
			mutable_cpu_data[index] = state[i][j];
			index++;
		}
	}

	target ? target_net->ForwardFromTo(0, last_layer_actor) : net->ForwardFromTo(0, last_layer_actor);

	const float* cpu_data = target ? target_blob_action->cpu_data() : blob_action->cpu_data();
	index = 0;

	//If the input batch was not full, we do not need to output the predictions for the whole batch
	for (size_t i = 0; i < state.size(); ++i)
	{
		for (size_t j = 0; j < prediction[i].size(); ++j)
		{
			prediction[i][j] = cpu_data[index];
			index++;
		}
	}

	return prediction;
}

std::vector<float> NN_Agent::PredictCritic(const std::vector<std::vector<float> > &state, const std::vector<std::vector<float> > &action, bool target)
{
	std::vector<float> prediction(target ? target_blob_q->count() : blob_q->count());

	float* mutable_cpu_data = target ? target_blob_state->mutable_cpu_data() : blob_state->mutable_cpu_data();
	int index = 0;

	for (size_t i = 0; i < state.size(); ++i)
	{
		for (size_t j = 0; j < state[i].size(); ++j)
		{
			mutable_cpu_data[index] = state[i][j];
			index++;
		}
	}

	mutable_cpu_data = target ? target_blob_action->mutable_cpu_data() : blob_action->mutable_cpu_data();
	index = 0;

	for (size_t i = 0; i < action.size(); ++i)
	{
		for (size_t j = 0; j < action[i].size(); ++j)
		{
			mutable_cpu_data[index] = action[i][j];
			index++;
		}
	}

	target ? target_net->ForwardFrom(last_layer_actor + 1) : net->ForwardFrom(last_layer_actor + 1);

	std::copy(target ? target_blob_q->cpu_data() : blob_q->cpu_data(), target ? target_blob_q->cpu_data() + target_blob_q->count() : blob_q->cpu_data() + blob_q->count(), prediction.data());

	return prediction;
}

void NN_Agent::TrainCritic(const std::vector<std::vector<float> > &state, const std::vector<std::vector<float> > &action, const std::vector<float>& y_i)
{
	//Reset all gradients to 0
	net->ClearParamDiffs();

	//Forward pass
	std::copy(y_i.begin(), y_i.end(), blob_label_q->mutable_cpu_data());
	PredictCritic(state, action, false);

	//Backward pass on the critic part of the net
	net->BackwardFromTo(net->layers().size() - 1, last_layer_actor + 1);

	//Apply the gradients calculated during the backward pass (the actor weights are not updated cause their gradients is still 0)
	solver->ApplyUpdate();
}

std::vector<float> NN_Agent::GetCriticGradient(const std::vector<std::vector<float> > &state, const std::vector<std::vector<float> > &action)
{
	//Forward, set output gradient to '1', Backward, get gradient w.r.t. the action
	net->ClearParamDiffs();
	PredictCritic(state, action, false);

	std::vector<float> ones = std::vector<float>(blob_q->count(), 1.0f);
	std::copy(ones.begin(), ones.end(), blob_q->mutable_cpu_diff());

	net->BackwardFromTo(net->layers().size() - 2, last_layer_actor + 1);

	std::vector<float> output(blob_action->count());

	std::copy(blob_action->cpu_diff(), blob_action->cpu_diff() + blob_action->count(), output.begin());

	return output;
}

void NN_Agent::TrainActor(const std::vector<std::vector<float> > &state, const std::vector<float>& grads)
{
	//Reset all the gradients to 0
	net->ClearParamDiffs();
	
	//Forward pass
	PredictActor(state, false);

	//Set the gradients w.r.t. the actions
	std::copy(grads.begin(), grads.end(), blob_action->mutable_cpu_diff());
	
	//Backward pass and update of the weights
	net->BackwardFrom(last_layer_actor);
	solver->ApplyUpdate();
}

void NN_Agent::UpdateTargetNets()
{
	if (target_net)
	{
		std::vector<std::string> source_layers = net->layer_names();

		boost::shared_ptr<caffe::Layer<float> > src_layer;
		boost::shared_ptr<caffe::Layer<float> > dst_layer;

		boost::shared_ptr<caffe::Blob<float> > src_blob;
		boost::shared_ptr<caffe::Blob<float> > dst_blob;

		for (size_t i = 0; i < source_layers.size(); ++i)
		{
			src_layer = net->layer_by_name(source_layers[i]);
			dst_layer = target_net->layer_by_name(source_layers[i]);

			//If the corresponding layer exists in the target net
			//and if they have the same number of parameters
			if (dst_layer != nullptr && src_layer->blobs().size() == dst_layer->blobs().size())
			{
				for (size_t j = 0; j < src_layer->blobs().size(); ++j)
				{
					src_blob = src_layer->blobs()[j];
					dst_blob = dst_layer->blobs()[j];

					if (src_blob->count() == dst_blob->count())
					{
						caffe::caffe_cpu_axpby(src_blob->count(), update_rate, src_blob->cpu_data(), 1.0f - update_rate, dst_blob->mutable_cpu_data());
					}
				}
			}
		}
	}
}