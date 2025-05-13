# AWSSimulations package

This repository has examples of two different ways to simulate the AWS interface.

1. [LocalStack][localstack]
2. [Moto][moto]

[LocalStack][localstack] is used to emulate the AWS CLI whereas [Moto][moto] is used for testing/mocking 
the AWS Python interface [Boto3][boto]. 


## Installation

The repository has been tested with python 3.11.11.

### Moto

```sh
pip install -r requirements.txt
``` 
#### Flash attention installation

```sh
pip install flash-attn --no-build-isolation
```

### Localstack

It is recommended to install this on a cloud desktop which has docker preinstalled. 

1. Follow the installation instructions on [LocalStack][localstack].
2. Setup a CDK files following this [example][example].
3. Update `bin` and `lib` folders with the example CDKs in this repository.


## Running the examples

Once the libraries are installed you can test the different simulators following the procedure below.

**Moto with CLI (recommended)**: `python envs/moto_cli_env.py`

**Moto**: `python envs/moto_env.py`

**Localstack**: `python envs/localstack_env.py`


## Learning Algorithms

The repository consists of several learning algorithms taken from the [Archer][archer] paper.
The code is taken from the [official git repository of Archer][archer_code] and cleaned + adapted
slightly for readability purposes.


## Running script from the terminal
`PYTHONPATH=path/to/AWSSimulations python your_script.py`

## Running experiments
1. [Run supervised fine-tuning][sft]
2. [Run online RL experiments][online_rl]
3. [Run offline RL experiments][offline_rl] (this is WIP and we have not tested offline RL code thoroughly yet.)



[localstack]: https://github.com/localstack/localstack
[moto]: https://github.com/getmoto/moto
[boto]: https://github.com/boto/boto3
[example]: https://docs.localstack.cloud/user-guide/integrations/aws-cdk/
[archer]: https://arxiv.org/pdf/2402.19446
[archer_code]: https://github.com/YifeiZhou02/ArCHer/tree/master
[sft]: scripts/supervised_fine_tuning/INFO.md
[online_rl]: scripts/online_learning/INFO.md
[offline_rl]: scripts/offline_learning/INFO.md