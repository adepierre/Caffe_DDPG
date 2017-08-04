#include "NN_Agent.h"

#include <chrono>
#include <algorithm>
#include <numeric>
#include <random>

NN_Agent::NN_Agent(const std::string &solver_actor_,
				   const std::string & solver_critic_, 
				   const float & tau_, 
				   const float & gamma_):
			update_rate(tau_),
			gamma(gamma_)
{
	//Create caffe objects (solver + net)
	caffe::SolverParameter solver_param_actor;
	caffe::ReadProtoFromTextFileOrDie(solver_actor_, &solver_param_actor);

	solver_actor.reset(caffe::SolverRegistry<float>::CreateSolver(solver_param_actor));

	net_actor = solver_actor->net();

	caffe::SolverParameter solver_param_critic;
	caffe::ReadProtoFromTextFileOrDie(solver_critic_, &solver_param_critic);

	solver_critic.reset(caffe::SolverRegistry<float>::CreateSolver(solver_param_critic));

	net_critic = solver_critic->net();


	caffe::NetParameter net_param_actor;
	net_actor->ToProto(&net_param_actor);

	net_param_actor.mutable_state()->set_phase(caffe::Phase::TEST);

	target_net_actor.reset(new caffe::Net<float>(net_param_actor));

	target_net_actor->CopyTrainedLayersFrom(net_param_actor);

	caffe::NetParameter net_param_critic;
	net_critic->ToProto(&net_param_critic);

	net_param_critic.mutable_state()->set_phase(caffe::Phase::TEST);

	target_net_critic.reset(new caffe::Net<float>(net_param_critic));

	target_net_critic->CopyTrainedLayersFrom(net_param_critic);

	//Save input and output blobs to find them easily later
	blob_state_actor = net_actor->blob_by_name("state");
	blob_state_critic = net_critic->blob_by_name("state");
	blob_action_actor = net_actor->blob_by_name("action");
	blob_action_critic = net_critic->blob_by_name("action");
	blob_q = net_critic->blob_by_name("q_values");
	blob_label_q = net_critic->blob_by_name("q_label");

	target_blob_state_actor = target_net_actor->blob_by_name("state");
	target_blob_action_actor = target_net_actor->blob_by_name("action");
	target_blob_state_critic = target_net_critic->blob_by_name("state");
	target_blob_action_critic = target_net_critic->blob_by_name("action");
	target_blob_q = target_net_critic->blob_by_name("q_values");
}

NN_Agent::NN_Agent(const std::string &model_file_actor, 
				   const std::string &model_file_critic,
				   const std::string &trained_file_actor,
				   const std::string &trained_file_critic)
{
	net_actor.reset(new caffe::Net<float>(model_file_actor, caffe::TEST));

	if (!trained_file_actor.empty())
	{
		net_actor->CopyTrainedLayersFrom(trained_file_actor);
	}

	net_critic.reset(new caffe::Net<float>(model_file_critic, caffe::TEST));

	if (!trained_file_critic.empty())
	{
		net_critic->CopyTrainedLayersFrom(trained_file_critic);
	}

	blob_state_actor = net_actor->blob_by_name("state");
	blob_action_actor = net_actor->blob_by_name("action");
	blob_state_critic = net_critic->blob_by_name("state");
	blob_action_critic = net_critic->blob_by_name("action");
	blob_q = net_critic->blob_by_name("q_values");
}

NN_Agent::~NN_Agent()
{
	if (solver_actor)
	{
		solver_actor->Snapshot();
	}

	if (solver_critic)
	{
		solver_critic->Snapshot();
	}
}

void NN_Agent::Train(const std::vector<std::vector<float> > &states, const std::vector<std::vector<float> > &actions, const std::vector<float> &rewards, const std::vector<bool> &terminals, const std::vector<std::vector<float>> &states_after)
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

	if (solver_actor->iter() % 100 == 0)
	{
		float norm = 0;
		int param_number = 0;
		for (size_t i = 0; i < net_actor->learnable_params().size(); i++)
		{
			for (size_t j = 0; j < net_actor->learnable_params()[i]->count(); j++)
			{
				float data = net_actor->learnable_params()[i]->cpu_data()[j];
				norm += data * data;
				param_number++;
			}
		}

		for (size_t i = 0; i < net_critic->learnable_params().size(); i++)
		{
			for (size_t j = 0; j < net_critic->learnable_params()[i]->count(); j++)
			{
				float data = net_critic->learnable_params()[i]->cpu_data()[j];
				norm += data * data;
				param_number++;
			}
		}

		norm = sqrt(norm) / param_number;
		std::cout << "Mean L2 norm of the trainable layers of the nets: " << norm << std::endl;
	}

	UpdateTargetNets();
}

std::vector<std::vector<float> > NN_Agent::PredictActor(const std::vector<std::vector<float> > &state, bool target)
{
	//A vector of batch size vectors of action dim size
	std::vector<std::vector<float> > prediction(target ? target_blob_action_actor->num() : blob_action_actor->num(), std::vector<float>(target ? target_blob_action_actor->count() / target_blob_action_actor->num() : blob_action_actor->count() / blob_action_actor->num(), 0.0f));

	float* mutable_cpu_data = target ? target_blob_state_actor->mutable_cpu_data() : blob_state_actor->mutable_cpu_data();
	int index = 0;

	for (size_t i = 0; i < state.size(); ++i)
	{
		for (size_t j = 0; j < state[i].size(); ++j)
		{
			mutable_cpu_data[index] = state[i][j];
			index++;
		}
	}

	target ? target_net_actor->Forward() : net_actor->Forward();

	const float* cpu_data = target ? target_blob_action_actor->cpu_data() : blob_action_actor->cpu_data();
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

	float* mutable_cpu_data = target ? target_blob_state_critic->mutable_cpu_data() : blob_state_critic->mutable_cpu_data();
	int index = 0;

	for (size_t i = 0; i < state.size(); ++i)
	{
		for (size_t j = 0; j < state[i].size(); ++j)
		{
			mutable_cpu_data[index] = state[i][j];
			index++;
		}
	}

	mutable_cpu_data = target ? target_blob_action_critic->mutable_cpu_data() : blob_action_critic->mutable_cpu_data();
	index = 0;

	for (size_t i = 0; i < action.size(); ++i)
	{
		for (size_t j = 0; j < action[i].size(); ++j)
		{
			mutable_cpu_data[index] = action[i][j];
			index++;
		}
	}

	target ? target_net_critic->Forward() : net_critic->Forward();

	std::copy(target ? target_blob_q->cpu_data() : blob_q->cpu_data(), target ? target_blob_q->cpu_data() + target_blob_q->count() : blob_q->cpu_data() + blob_q->count(), prediction.data());

	return prediction;
}

void NN_Agent::TrainCritic(const std::vector<std::vector<float> > &state, const std::vector<std::vector<float> > &action, const std::vector<float>& y_i)
{
	//Reset all gradients to 0
	net_critic->ClearParamDiffs();

	//Forward pass
	std::copy(y_i.begin(), y_i.end(), blob_label_q->mutable_cpu_data());
	PredictCritic(state, action, false);

	net_critic->Backward();
	solver_critic->ApplyUpdate();
	solver_critic->iter_++;

	if (solver_critic->iter() > 0 && solver_critic->iter() % solver_critic->param().snapshot() == 0)
	{
		solver_critic->Snapshot();
	}
}

std::vector<float> NN_Agent::GetCriticGradient(const std::vector<std::vector<float> > &state, const std::vector<std::vector<float> > &action)
{
	//Forward, set output gradient to '1', Backward, get gradient w.r.t. the action
	PredictCritic(state, action, false);

	std::vector<float> ones = std::vector<float>(blob_q->count(), 1.0f);
	std::copy(ones.begin(), ones.end(), blob_q->mutable_cpu_diff());

	net_critic->BackwardFrom(net_critic->layers().size() - 2);

	std::vector<float> output(blob_action_critic->count());

	std::copy(blob_action_critic->cpu_diff(), blob_action_critic->cpu_diff() + blob_action_critic->count(), output.begin());

	return output;
}

void NN_Agent::TrainActor(const std::vector<std::vector<float> > &state, const std::vector<float>& grads)
{
	//Reset all the gradients to 0
	net_actor->ClearParamDiffs();
	
	//Forward pass
	PredictActor(state, false);

	//Set the gradients w.r.t. the actions
	std::copy(grads.begin(), grads.end(), blob_action_actor->mutable_cpu_diff());
	
	//Backward pass and update of the weights
	net_actor->Backward();
	solver_actor->ApplyUpdate();
	solver_actor->iter_++;

	if (solver_actor->iter() > 0 && solver_actor->iter() % solver_actor->param().snapshot() == 0)
	{
		solver_actor->Snapshot();
	}
}

void NN_Agent::UpdateTargetNets()
{
	if (target_net_actor)
	{
		std::vector<std::string> source_layers = net_actor->layer_names();

		boost::shared_ptr<caffe::Layer<float> > src_layer;
		boost::shared_ptr<caffe::Layer<float> > dst_layer;

		boost::shared_ptr<caffe::Blob<float> > src_blob;
		boost::shared_ptr<caffe::Blob<float> > dst_blob;

		for (size_t i = 0; i < source_layers.size(); ++i)
		{
			src_layer = net_actor->layer_by_name(source_layers[i]);
			dst_layer = target_net_actor->layer_by_name(source_layers[i]);

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

	if (target_net_critic)
	{
		std::vector<std::string> source_layers = net_critic->layer_names();

		boost::shared_ptr<caffe::Layer<float> > src_layer;
		boost::shared_ptr<caffe::Layer<float> > dst_layer;

		boost::shared_ptr<caffe::Blob<float> > src_blob;
		boost::shared_ptr<caffe::Blob<float> > dst_blob;

		for (size_t i = 0; i < source_layers.size(); ++i)
		{
			src_layer = net_critic->layer_by_name(source_layers[i]);
			dst_layer = target_net_critic->layer_by_name(source_layers[i]);

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