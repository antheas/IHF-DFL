# OpenFL: Decentralized Federated Learning on Public Blockchain Systems

```
 ___ _   _ ____       ____  _____ _     
|_ _| | | |  _ \     |  _ \|  ___| |    
 | || |_| | |_) |____| | | | |_  | |    
 | ||  _  |  __/_____| |_| |  _| | |___ 
|___|_| |_|_|        |____/|_|   |_____|                      
```

# Getting started
## 1. Ganache
- Download Ganache
- Set up a workspace (Not quickstart)
- Set gas limit much higher than default, same with balance
- Set accounts to 8

## 2. Environment Variables
The project contains a .env file located in the .env folder, but supports easy replacement of this environmnent.
The project runs with the .env.ganache .env file by default. If another .env is preferred run the program with the 
``ENV=<env_file_identifier>`` prefix. Providing no ENV prefix and providing ``ENV=ganache`` is therefore equivalent.

In your Environment, you must have the following variables set:
```
RPC_URL="<RPC_URL from ganache or sepolia, including port>"
PRIVATE_KEYS="<Private keys from your accounts colon separated (for non-locally forked blockchain). If you have fork=true (using Ganache), there is no need to set private keys. Then just keep this variable empty>"
```

## 3. Requirements

Python 3.10 or newer is supported. On Ubuntu, install the system tools with:

```bash
sudo apt update
sudo apt install python3 python3-venv nodejs npm curl
```

`run_all.sh` creates `.venv` with the local `python3` and installs the runtime
dependencies automatically. It also installs the pinned Ganache and Solidity command-line
tools locally under `node_modules`. Before launching parallel workers, it downloads each
selected dataset once to avoid concurrent writes to the same archive. To prepare the
software environment without starting the experiments:

```bash
./run_all.sh --setup-only
```

Set `PYTHON=/path/to/python` on the first run to use another local interpreter, or
set `VENV_DIR=/path/to/venv` to put the environment elsewhere. The script always
runs through the venv, so it does not need to be activated.

For development tools, activate the environment and install the optional dependencies:

```bash
source .venv/bin/activate
python -m pip install -e ".[dev]"
```

Build the abi and bytecode files from the smart contracts
``.venv/bin/python scripts/compile_contracts.py``

Strip notebook outputs before committing (run once per clone):
``nbstripout --install``

Install ROCM(AMD) or CUDA(NVIDIA):
ROCM:
``.venv/bin/python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm7.1``
CUDA:
``.venv/bin/python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130``


## 4. Running an Experiment

Two modes are availble for running an experiment: sample or experiment.
### Sample
Just want to see a quick demo of the code? Run the sample.py file with:
`python ./experiment/sample.py`

### Experiment
Want to run a systematic experiment with different parameters?

The Experiment folder contains files for running experiments on different datasets.
To change the experiment setup, modify the experiment_configuration.py file.
To change the dataset, modify the experiments.py file.

The file experiments.py runs one such experiment and can be run with:

`python ./experiment/experiments.py`

Or if you want to specify a different .env file, run with the ENV prefix as described in the Environment Variables section:
``ENV=ganache python ./experiment/experiments.py``

To run several experiments at the same time, pass ``--parallel N``:
``python ./experiment/experiments.py --parallel 4``

Each run gets its own process, and its console output goes to a ``.log`` file next to its ``.csv``.
All runs share the same Ganache node and accounts (each deploys its own contracts), so make sure the
account balances are large enough for N runs at once. Parallel mode only supports ``fork=True``.


## 5. Solidity testing
- All forge commands must be run in the foundry directory.
- Download Foundry in a Unix-like shell (WSL or Linux): 
  - `curl -L https://foundry.paradigm.xyz | bash && source ~/.bashrc`
  - `foundryup`
  - `cd foundry && forge soldeer install`
  
- To Test:
  - `cd foundry`
  - `forge build` 
  - `forge test`

# 6. Running Tests

Tests are run with pytest from the repo root:
```
pytest
```

By default, tests marked as `slow` are skipped. These are tests that load full datasets (e.g. CIFAR-10)
and can take significant time. To include them:
```
pytest -m slow
```

To run only the slow tests:
```
pytest -m slow --no-header -q
```

Solidity tests (forge) run automatically as part of the pytest suite on Linux/WSL.
On Windows they are skipped — run them manually in WSL:
```
cd foundry && forge test
```

# 7. Test Coverage

To get test coverage for python code, run the following command: \
`pytest --cov tests/`

To get test coverage for solidity code, run the following command: \
`forge coverage`

(Optional) Output to a file
`forge coverage --report lcov`


# 8. Solidity Compiler (Arm)

On ARM Linux, `run_all.sh` uses the architecture-independent Solidity 0.8.9
compiler installed from npm. No separate native `solc` download is required.
