"../bin/Release/Caffe_DDPG.exe" ^
--train=0 ^
--display=1 ^
--FPS=50 ^
--model_actor=net_actor.prototxt ^
--model_critic=net_critic.prototxt ^
--weights_actor=Trained_Actor.caffemodel ^
--weights_critic=Trained_Critic.caffemodel ^
--test_episode=1