# Run Aloha Sim

## Run

Terminal window 1:

```bash
# Create virtual environment
conda create -n pi-conda python=3.10 examples/aloha_sim/pi-conda -y
source examples/aloha_sim/pi-conda/bin/activate
python -m pip install -r examples/aloha_sim/requirements.txt
python -m pip install -e packages/openpi-client

# Run the simulation
MUJOCO_GL=egl python examples/aloha_sim/main.py
```

Note: If you are seeing EGL errors, you may need to install the following dependencies:

```bash
sudo apt-get install -y libegl1-mesa-dev libgles2-mesa-dev
```

Terminal window 2:

```bash
# Run the server
conda run -n pi-conda python scripts/serve_policy.py --env ALOHA_SIM
```
