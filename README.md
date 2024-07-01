# Isaac Sim Benchmarking
Using nvidia Isaac Sim to test robotics simulation and robotics ML training using cloud GPUs

## Setup 

### Remove sudo requirement for running docker
```
export USERNAME=$(whoami)
sudo groupadd docker
sudo usermod -aG docker $USERNAME
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

### Pull the Isaac Sim container
Pull the Isaac Sim docker container
```
docker pull nvcr.io/nvidia/isaac-sim:4.0.0
```

### Launch Isaac Sim

Clone this repo if you haven't done so:

```
git clone https://github.com/LambdaLabsML/isaac-sim-benchmarking.git && \
cd isaac-sim-benchmarking
```

Run the Isaac Sim container with an interactive Bash session
```
docker run --name isaac-sim --entrypoint bash -it --gpus all -e "ACCEPT_EULA=Y" --rm --network=host \
    -e "PRIVACY_CONSENT=Y" \
    -v ~/docker/isaac-sim/cache/kit:/isaac-sim/kit/cache:rw \
    -v ~/docker/isaac-sim/cache/ov:/root/.cache/ov:rw \
    -v ~/docker/isaac-sim/cache/pip:/root/.cache/pip:rw \
    -v ~/docker/isaac-sim/cache/glcache:/root/.cache/nvidia/GLCache:rw \
    -v ~/docker/isaac-sim/cache/computecache:/root/.nv/ComputeCache:rw \
    -v ~/docker/isaac-sim/logs:/root/.nvidia-omniverse/logs:rw \
    -v ~/docker/isaac-sim/data:/root/.local/share/ov/data:rw \
    -v ~/docker/isaac-sim/documents:/root/Documents:rw \
    -v $(pwd)/scripts:/workspace:rw \
    -v $(pwd)/patch/nucleus.py:/isaac-sim/exts/omni.isaac.nucleus/omni/isaac/nucleus/nucleus.py:ro \
    nvcr.io/nvidia/isaac-sim:4.0.0
```
Notice we patch the `nucleus.py` script with the update [here](https://forums.developer.nvidia.com/t/detected-a-blocking-function-this-will-cause-hitches-or-hangs-in-the-ui-please-switch-to-the-async-version/271191/12) to avoid `world.scene.add_default_ground_plane()` from causing a `blocking function`. 

We also mount examples scripts in the `scripts` folder to `workspace` inside the container.

The next step is to launch Isaac Sim in the background using:
```
./runheadless.webrtc.sh &
```
This should allow for streaming UI to a webRTC client but this is a WIP [(docs here)](https://docs.omniverse.nvidia.com/extensions/latest/ext_livestream/webrtc.html)

You should see the text `Isaac Sim Headless Native App is loaded.` when it is finished loading.

Finally, run an example script:
```
# bash /isaac-sim/python.sh /workspace/my_application.py
```
This will print any logs from the simulation that are in the python script. e.g.

```
Cube's orientation is : [ 1.0000000e+00 -1.2853170e-06  5.4760726e-08  7.0710399e-08]
Cube's linear velocity is : [ 3.7621916e-04 -7.0683210e-04 -9.7330332e-05]
Cube position is : [-0.0004345   0.00036895  0.25074962]
Cube's orientation is : [ 1.0000000e+00 -1.3024535e-06  2.0566535e-08  5.3247195e-08]
Cube's linear velocity is : [ 3.7617341e-04 -7.0690154e-04 -9.7553268e-05]
Cube position is : [-0.00043449  0.00036895  0.2507496 ]
Cube's orientation is : [ 1.0000000e+00 -1.2903952e-06  7.4222413e-08  5.0700635e-08]
Cube's linear velocity is : [ 3.7670622e-04 -7.0628989e-04 -9.7703909e-05]
...
[19.301s] Simulation App Shutting Down
```