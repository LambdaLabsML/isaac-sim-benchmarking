# Isaac Sim Benchmarking
Using nvidia Isaac Sim to test robotics simulation and robotics ML training using cloud GPUs

## Setup 
[(Referencing this instalation guide)](https://docs.omniverse.nvidia.com/isaacsim/latest/installation/install_container.html)

### Install the NVIDIA Container Toolkit
Configure the repository
```
$ curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
  && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list \
  && \
    sudo apt-get update
```
Install the NVIDIA Container Toolkit packages
```
$ sudo apt-get install -y nvidia-container-toolkit
$ sudo systemctl restart docker
```
Configure the container runtime
```
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```
Verify NVIDIA Container Toolkit
```
sudo docker run --rm --runtime=nvidia --gpus all ubuntu nvidia-smi
```

### Log into nvidia NGC to get access to Isaac Sim
Create/log into a nvidia NGC account [here](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/isaac-sim)

Generate your [NGC API Key](https://docs.nvidia.com/ngc/gpu-cloud/ngc-user-guide/index.html#generating-api-key)

Log in to NGC on the GPU instance
```
$ sudo docker login nvcr.io
Username: $oauthtoken
Password: <Your NGC API Key>
WARNING! Your password will be stored unencrypted in /home/username/.docker/config.json.
Configure a credential helper to remove this warning. See
https://docs.docker.com/engine/reference/commandline/login/#credentials-store
Login Succeeded
```
You can reuse your API key for future setups

### Pull and run the Isaac Sim container
Pull the Isaac Sim docker container
```
$ sudo docker pull nvcr.io/nvidia/isaac-sim:4.0.0
```
Run the Isaac Sim container with an interactive Bash session
```
$ sudo docker run --name isaac-sim --entrypoint bash -it --runtime=nvidia --gpus all -e "ACCEPT_EULA=Y" --rm --network=host \
    -e "PRIVACY_CONSENT=Y" \
    -v ~/docker/isaac-sim/cache/kit:/isaac-sim/kit/cache:rw \
    -v ~/docker/isaac-sim/cache/ov:/root/.cache/ov:rw \
    -v ~/docker/isaac-sim/cache/pip:/root/.cache/pip:rw \
    -v ~/docker/isaac-sim/cache/glcache:/root/.cache/nvidia/GLCache:rw \
    -v ~/docker/isaac-sim/cache/computecache:/root/.nv/ComputeCache:rw \
    -v ~/docker/isaac-sim/logs:/root/.nvidia-omniverse/logs:rw \
    -v ~/docker/isaac-sim/data:/root/.local/share/ov/data:rw \
    -v ~/docker/isaac-sim/documents:/root/Documents:rw \
    nvcr.io/nvidia/isaac-sim:4.0.0
```

### Launch Isaac Sim
Launch Isaac Sim in the background using:
```
./runheadless.webrtc.sh &
```
This should allow for streaming UI to a webRTC client but this is a WIP [(docs here)](https://docs.omniverse.nvidia.com/extensions/latest/ext_livestream/webrtc.html)

You should see the text `Isaac Sim Headless Native App is loaded.` when it is finished loading.

### Create standalone python script
Install a text editor like vi
```
# apt-get update
# apt-get install vim
```
Create a new python file. To follow the [nvidia docs example here](https://docs.omniverse.nvidia.com/isaacsim/latest/core_api_tutorials/tutorial_core_hello_world.html#converting-the-example-to-a-standalone-application), create a new file called `my_application.py` under `/isaac-sim/exts/omni.isaac.examples/omni/isaac/examples/user_examples/` and add the following code:
```python
#launch Isaac Sim before any other imports
#default first two lines in any standalone application
from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": True}) # we can also run as headless.

from omni.isaac.core import World
from omni.isaac.core.objects import DynamicCuboid
import numpy as np

world = World()
world.scene.add_default_ground_plane()
fancy_cube =  world.scene.add(
    DynamicCuboid(
        prim_path="/World/random_cube",
        name="fancy_cube",
        position=np.array([0, 0, 1.0]),
        scale=np.array([0.5015, 0.5015, 0.5015]),
        color=np.array([0, 0, 1.0]),
    ))
# Resetting the world needs to be called before querying anything related to an articulation specifically.
# Its recommended to always do a reset after adding your assets, for physics handles to be propagated properly
world.reset()
for i in range(100):
    position, orientation = fancy_cube.get_world_pose()
    linear_velocity = fancy_cube.get_linear_velocity()
    # will be shown on terminal
    print("Cube position is : " + str(position))
    print("Cube's orientation is : " + str(orientation))
    print("Cube's linear velocity is : " + str(linear_velocity))
    # we have control over stepping physics and rendering in this workflow
    # things run in sync
    world.step(render=True) # execute one physics step and one rendering step

simulation_app.close() # close Isaac Sim
```
This creates a cube 1 unit in the air and logs its position as it falls to the ground for 100 steps.

If you create your own standalone python script, make sure to use the first two lines as it will not work without them.
```python
from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": True})
```

When running the simulation, there seems to be an issue where using `world.scene.add_default_ground_plane()` causes a `blocking function`. [A forum post detailed a similar issue and had a fix](https://forums.developer.nvidia.com/t/detected-a-blocking-function-this-will-cause-hitches-or-hangs-in-the-ui-please-switch-to-the-async-version/271191/12). 

Edit the `exts/omni.isaac.nucleus/omni/isaac/nucleus/nucleus.py` file 
```
# vi exts/omni.isaac.nucleus/omni/isaac/nucleus/nucleus.py
```
and change the `check_server` function to below:
```python
def check_server(server: str, path: str, timeout: float = 10.0) -> bool:
    """Check a specific server for a path

    Args:
        server (str): Name of Nucleus server
        path (str): Path to search

    Returns:
        bool: True if folder is found
    """
    carb.log_info("Checking path: {}{}".format(server, path))
    # Increase hang detection timeout
    if "localhost" not in server:
        omni.client.set_hang_detection_time_ms(10000)
        result, _ = omni.client.stat("{}{}".format(server, path))
        if result == Result.OK:
            carb.log_info("Success: {}{}".format(server, path))
            return True
    carb.log_info("Failure: {}{} not accessible".format(server, path))
    return False
```
This should fix the issue.

Finally, run the python script
```
# ./python.sh ./exts/omni.isaac.examples/omni/isaac/examples/user_examples/my_application.py
```
This will print any logs from the simulation that are in the python script.