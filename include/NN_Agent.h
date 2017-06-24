#pragma once

#include <string>
#include <vector>
#include <memory>
#include <random>

#include <caffe/caffe.hpp>


class NN_Agent
{
public:
	/**
	* \brief Create an agent for training, initialize net and solver
	* \param solver_ Caffe solver (*.prototxt file)
	* \param tau_ Soft target update rate  tau*net + (1-tau)*target
	* \param gamma_ Reward discount factor ( R = reward_t + gamma * Q_t+1)
	*/
	NN_Agent(const std::string &solver_,
			 const float &tau_,
			 const float &gamma_);

	/**
	* \brief Create an agent for testing, initialize the net
	* \param model_file Caffe model file to generate the net (*.prototxt)
	* \param trained_file Caffe caffemodel file to initialize weights (*.caffemodel)
	*/
	NN_Agent(const std::string &model_file,
			 const std::string &trained_file);

	~NN_Agent();

	/**
	* \brief Perform one train cycle on the whole net (actor+critic)
	* \param states One batch of states
	* \param actions Corresponding batch of actions
	* \param rewards Corresponding batch of rewards
	* \param terminals Batch of booleans, true if the state is terminal
	* \param states_after Batch of states after the action has been performed
	*/
	void Train(const std::vector<std::vector<float> > &states, const std::vector<std::vector<float> > &actions, const std::vector<float> &rewards, const std::vector<bool> &terminals, const std::vector<std::vector<float> > &states_after);

	/**
	* \brief Return the predicted action given a state
	* \param state The current states, as a batch of vectors of floats
	* \param target Whether we want the prediction of the real net or the target net
	* \return The predicted actions, as a batch of vectors of floats
	*/
	std::vector<std::vector<float> > PredictActor(const std::vector<std::vector<float> > &state, bool target);

	/**
	* \brief Return the predicted Q value given a state and an action
	* \param state The current states, as a batch of vectors of floats
	* \param action The actions
	* \param target Whether we want the prediction of the real net or the target net
	* \return The predicted Q value, as a vector of batch size floats
	*/
	std::vector<float> PredictCritic(const std::vector<std::vector<float> > &state, const std::vector<std::vector<float> > &action, bool target);

	/**
	* \brief Perform one step on the critic net
	* \param state The current states, as a batch of vector of floats
	* \param action The actions
	* \param y_i The target value (for example ri + gamma * ri+1)
	*/
	void TrainCritic(const std::vector<std::vector<float> > &state, const std::vector<std::vector<float> > &action, const std::vector<float> &y_i);

	/**
	* \brief Get the gradient of the critic net w.r.t the action
	* \param state The current states, as a batch of vectors of floats
	* \param action The actions
	* \return The gradient
	*/
	std::vector<float> GetCriticGradient(const std::vector<std::vector<float> > &state, const std::vector<std::vector<float> > &action);

	/**
	* \brief Perform one step on the actor net
	* \param state The current state, as a batch of vectors of floats
	* \param grads The gradients given by the critic net
	*/
	void TrainActor(const std::vector<std::vector<float> > &state, const std::vector<float> &grads);

	/**
	* \brief Update the target networks weights
	*/
	void UpdateTargetNets();

protected:

	//Common parameters
	boost::shared_ptr<caffe::Net<float> > net;

	boost::shared_ptr<caffe::Blob<float> > blob_state;
	boost::shared_ptr<caffe::Blob<float> > blob_action;
	boost::shared_ptr<caffe::Blob<float> > blob_q;
		
	//Parameters used for training
	boost::shared_ptr<caffe::Solver<float> > solver;

	boost::shared_ptr<caffe::Blob<float> > blob_label_q;

	boost::shared_ptr<caffe::Net<float> > target_net;

	boost::shared_ptr<caffe::Blob<float> > target_blob_state;
	boost::shared_ptr<caffe::Blob<float> > target_blob_action;
	boost::shared_ptr<caffe::Blob<float> > target_blob_q;

	float update_rate;
	float gamma;

	int last_layer_actor;
};


