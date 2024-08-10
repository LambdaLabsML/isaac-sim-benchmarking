# Isaac Sim Benchmarking
Using nvidia Isaac Sim and Isaac Lab to test robotics simulation and robotics ML training using cloud GPUs

## Setup 

### Remove sudo requirement for running docker
```
export USERNAME=$(whoami) && \
sudo usermod -aG docker $USERNAME && \
newgrp docker
```
### Install the NVIDIA Container Toolkit

NVIDIA Container Toolkit is installed by default on [Lambda Cloud](https://lambdalabs.com/). To verify, log into your instance and run this and see it prints out GPU info successfully.

```
docker run --rm --gpus all ubuntu nvidia-smi
```

If you need to install it on your own machine, here are the [steps](https://docs.omniverse.nvidia.com/isaacsim/latest/installation/install_container.html)


### Log into nvidia NGC to get access to Isaac Sim
Create/log into a nvidia NGC account [here](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/isaac-sim)

Generate your [NGC API Key](https://docs.nvidia.com/ngc/gpu-cloud/ngc-user-guide/index.html#generating-api-key)

Log in to NGC on the GPU instance
```
$ docker login nvcr.io
Username: $oauthtoken
Password: <Your NGC API Key>
WARNING! Your password will be stored unencrypted in /home/username/.docker/config.json.
Configure a credential helper to remove this warning. See
https://docs.docker.com/engine/reference/commandline/login/#credentials-store
Login Succeeded
```
You can reuse your API key for future setups

### Launch Isaac Lab Docker Container
Clone the Isaac Lab repo if you haven't done so already:

```
git clone https://github.com/RajitGhosh/IsaacLab-Fork.git && \
cd IsaacLab-Fork
```
Launch the container in a detached state and enter it:
```bash
# Launch the container in detached mode
# We don't pass an image extension arg, so it defaults to 'base'
./docker/container.sh start
# Enter the container
# We pass 'base' explicitly, but if we hadn't it would default to 'base'
./docker/container.sh enter base
```
To copy files from the base container to the host machine, you can use the following command:
```bash
# Copy the file /workspace/isaaclab/logs to the current directory
docker cp isaac-lab-base:/workspace/isaaclab/logs .
```

## Running Basic Scripts
The easiest way to run a script through a docker container is with headless mode (meaning no GUI is displayed). This can be used to efficiently train reinforcement learning models.

To run any script headlessly without livestreaming the UI, append `--headless` or set the `HEADLESS` environment variable to 1. For example:
```bash
isaaclab -p source/standalone/tutorials/00_sim/log_time.py --headless
# OR
export HEADLESS=1
isaaclab -p source/standalone/tutorials/00_sim/log_time.py
```

This will produce logs in the log file `/workspace/isaaclab/logs/docker_tutorial` inside the container, which can be retrieved by exiting the docker container and running the following in the `IsaacLab` directory:
```
./docker/container.sh copy
```
The logs will be added to `/isaaclab/docker/artifacts/logs/docker_tutorial` in the host terminal environment.
(see [Isaac Lab docs](https://isaac-sim.github.io/IsaacLab/source/deployment/run_docker_example.html#executing-the-script) for more info)

## Training a Model
Isaac Lab supports 4 different reinforcement learning libraries:
- Stable-Baselines3
- SKRL
- RL-Games
- RSL-RL

To train an SKRL model on the provided Isaac Cartpole task, for example, run the following inside of the container:
```bash
# install python module (for skrl)
./isaaclab.sh -i skrl
# run script for training
./isaaclab.sh -p source/standalone/workflows/skrl/train.py --task Isaac-Cartpole-v0 --headless
```
When finished, this will add checkpoints of the best agents during training to `/workspace/isaaclab/logs/skrl/cartpole/<date-time>/` and save logs to Tensorboard.

To see the full list of provided tasks, run the following command or navigate to the [environments list page](https://isaac-sim.github.io/IsaacLab/source/features/environments.html):
```
./isaaclab.sh -p source/standalone/environments/list_envs.py
```
Again, more info can be found on the [official docs](https://isaac-sim.github.io/IsaacLab/source/setup/sample.html#reinforcement-learning).

### Viewing Training Logs
All of the training scripts log training progress and other details to a tensorboard log file inside the `logs` folder. Since the instance doesn't accept your local computer's IP, you have to use port forwarding with an ssh command. 

If you don't yet have an ssh key pair, you can create one using the `ssh-keygen` command.

First, go into your instances `~/.ssh/authorized_keys` file and add your public ssh key. You can then connect to your instance through ssh:
```
ssh -N -f -L localhost:16006:localhost:6006 -i <path to private key> ubuntu@<instance IP>
```
This forwards the default tensorboard port of 6006 to the port 16006 on your local computer.

Finally, run the following command to launch tensorboard, and open http://localhost:16006. This will direct you to the tensorboard logs page.
```bash
# execute from the root directory of the repository
./isaaclab.sh -p -m tensorboard.main --logdir=logs
```

### Multi GPU Training
Training with multiple GPUs is also possible, but only with the RL-Games and SKRL libraries. To train with multiple GPUs on RL-Games, use the following command, where --proc_per_node represents the number of available GPUs (e.g. 8 on an 8xA100):
```bash
python -m torch.distributed.run --nnodes=1 --nproc_per_node=8 source/standalone/workflows/rl_games/train.py --task=Isaac-Cartpole-v0 --headless --distributed
```
See the [official docs](https://isaac-sim.github.io/IsaacLab/source/features/multi_gpu.html) for more info.

## Viewing a Model
To view the model performing the task, you can either record and view an mp4 video of the model during or after training (recommended) or livestream Isaac Sim GUI to a remote client from an RTX instance.

### Video Recording
The easiest way to view a model during training or playback is with a video recording. This works on all types of GPUs and does not require an external client.

#### Training
To enable video recording during training, use the `--video` flag when running the training script:
```bash
./isaaclab.sh -p source/standalone/workflows/skrl/train.py --task Isaac-Cartpole-v0 --headless --video
```
Additionally, use the `--video_length` (length of each video in time steps) and `--video_interval` (interval between video recordings in time steps) flags to control parameters related to recording the video. 

This will add periodic mp4 videos of the model as it goes through the training process to the logs file. 

#### Playback
Recording videos of training causes the simulation to spend resources on rendering and recording, thus slowing down the training. To combat this, the forked version of IsaacLab adds support for recording a video while doing playback of an SKRL, RL-Games, or RSL-RL model. Video recording is enabled on default, and can be disabled with the `--disable_video` flag. For example, to play the model we previously trained at any checkpoint, use the following (with the path being to the desired checkpoint):
```bash
./isaaclab.sh -p source/standalone/workflows/skrl/play.py --task Isaac-Reach-Franka-v0 --num_envs 32 --headless --checkpoint /PATH/TO/model.pt
```
You can again use the `--video_length` parameter to control the length of the clip.

Since this is still a work in progress, the easiest way to retrieve the recording is to wait a few minutes (depending on the length of the video) and then shut down the playback with `ctrl-c`. 

#### Viewing the Video
To easily view the recorded video, exit the docker container and run the following in the `IsaacLab` directory:
```
./docker/container.sh copy
```
The logs will be added to `/isaaclab/docker/artifacts/logs/<library>/<task>/<date-time>` in the host terminal environment. The recorded videos are in the `videos` folder, and can be downloaded and viewed. If you see a `.json` file but not an `mp4`, try letting the video training or playback run longer before terminating.

### Livestream
You can also stream the UI to an external client using the `--livestream` flag. The easiest way to do this is through the native Omniverse Streaming Client, but this is only supported on Windows and Linux ([see this tutorial for installing the launcher](https://docs.omniverse.nvidia.com/launcher/latest/installing_launcher.html) and [this tutorial for setting up the streaming client](https://docs.omniverse.nvidia.com/streaming-client/latest/user-manual.html#installation-and-usage)). Alternatively, it should be possible to use WebRTC to do livestreaming ([although there are some known issues with it](https://isaac-sim.github.io/IsaacLab/source/deployment/docker.html#webrtc-streaming))

> [!IMPORTANT]  
> For livestreaming to work, the GPU instance must have an RTX card. This means that all agents trained on a non-RTX instance must be transferred to an RTX instance in order to view using this method.

Once the livestream client is set up, you can run an Isaac Lab script with `--livestream 1` (or `--livestream 2` if using WebRTC):
```
./isaaclab.sh -p source/standalone/demos/quadrupeds.py --livestream 1
```
Alternatively, you can set the LIVESTREAM environment variable:
```
export LIVESTREAM=1
./isaaclab.sh -p source/standalone/demos/quadrupeds.py
```
Once you see the message `Simulation App Startup Complete`, you can enter the IP of the hosting GPU instance into the streaming client and press `connect`. When loaded, you will see the simulation being run and rendered in real time.

This can also be used to view the result of a machine learning model. For example, you can play the model you already trained with SKRL:
```bash
# run script for playing with 32 environments
./isaaclab.sh -p source/standalone/workflows/skrl/play.py --task Isaac-Cartpole-v0 --num_envs 32 --checkpoint /workspace/isaaclab/logs/skrl/cartpole/<date-time>/checkpoints/best_agent.pt --livestream 1 --disable_video
```
Once connected to the simulation, you can view the best agent in the model you trained:
![](images/cartpole.png)
