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

### Runing scripts 
While in this state, there are two ways of running the simulation headlessly: with livestream, and without. 

#### No livestream:
To run an isaac lab script headlessly without livestreaming the UI, append `--headless` or set the `HEADLESS` environment variable to 1. For example:
```bash
isaaclab -p source/standalone/tutorials/00_sim/log_time.py --headless
# OR
export HEADLESS=1
isaaclab -p source/standalone/tutorials/00_sim/log_time.py
```
This will produce logs in the log file `/workspace/isaaclab/logs/docker_tutorial`, which can be retrieved by exiting the docker container, going to `IsaacLab`, and running:
```
./docker/container.sh copy
```
(see [isaac lab docs](https://isaac-sim.github.io/IsaacLab/source/deployment/run_docker_example.html#executing-the-script) for more info)

We can also use headless without livestreaming to train reinforcement learning models. For example:
```bash
# install python module (for skrl)
./isaaclab.sh -i skrl
# run script for training
./isaaclab.sh -p source/standalone/workflows/skrl/train.py --task Isaac-Cartpole-v0 --headless
```
When finished, this will add checkpoints of the best agents during training to `/workspace/isaaclab/logs/skrl/cartpole/<date-time>/` and save logs to Tensorboard.
Again, more info can be found on the [official docs](https://isaac-sim.github.io/IsaacLab/source/setup/sample.html#reinforcement-learning).

#### Livestream:
We can also stream the UI to an external client using the `--livestream` flag. The easiest way to do this is through the native Omniverse Streaming Client, but this is only supported on Windows and Linux ([see this tutorial for installing the launcher](https://docs.omniverse.nvidia.com/launcher/latest/installing_launcher.html) and [this tutorial for setting up the streaming client](https://docs.omniverse.nvidia.com/streaming-client/latest/user-manual.html#installation-and-usage)). Alternatively, it should be possible to use WebRTC to do livestreaming ([although there are some known issues with it](https://isaac-sim.github.io/IsaacLab/source/deployment/docker.html#webrtc-streaming))

> [!IMPORTANT]  
> For livestreaming to work, the GPU instance must have an RTX card.

Once the livestream client is set up, we can run an isaac lab script with `--livestream 1` (or `--livestream 2` if using WebRTC):
```
./isaaclab.sh -p source/standalone/demos/quadrupeds.py --livestream 1
```
Alternatively, we can set the LIVESTREAM environment variable:
```
export LIVESTREAM=1
./isaaclab.sh -p source/standalone/demos/quadrupeds.py
```
Once we see the message `Simulation App Startup Complete`, we can enter the IP of the hosting GPU instance into the streaming client and press `connect`. When loaded, we will see the simulation being run and rendered in real time.

This can also be used to view the result of a machine learning model. For example, we can play the model we already trained with SKRL:
```bash
# run script for playing with 32 environments
./isaaclab.sh -p source/standalone/workflows/skrl/play.py --task Isaac-Cartpole-v0 --num_envs 32 --checkpoint /workspace/isaaclab/logs/skrl/cartpole/<date-time>/checkpoints/best_agent.pt --livestream 1
```
Once connected to the simulation, we can view the best agent in the model we trained:
![](images/cartpole.png)
