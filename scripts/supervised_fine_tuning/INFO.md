## Supervised fine-tuning Experiment Setup

The ``sft.py`` file loads a dataset of state-action tuples $(s_t, a_t)$ and fine-tunes an LLM/policy to map the
state $s_t$ to the action $a_t$.
During the training, we let the model interact with the AWS environment and evaluate its performance.
The folder ``logs`` is created where the trained checkpoints are saved. Furthermore, we store the history of interactions
between the agent and the environment in the logs.

### Running the experiments

1. Without accelerate: ``python sft.py``
2. With accelerate: ``accelerate launch --config_file config/accelerator_config.yaml sft.py``

### Key Considerations

1. When running multiple seeds on the same node/compute instance, make sure that the port number for the AWS environment
   is unique per run. This is crucial to avoid overlap in the environments from different runs. The seed is unique per
   run and we specify a unique port number for the moto server for each run. When we execute another run of the
   algorithm in parallel we have to make sure that the port where moto is deployed is empty. We can pass the port number
   to the python script as an argument or change it in the ``config/bc/aws.yaml`` file.
2. You can change the hyperparameters during training, e.g., learning rate, from ``config/bc/aws.yaml`` and
   ``config/bc/default.yaml``. ``config/bc/aws.yaml`` overrides some hyperparameters from ``config/bc/default.yaml``.
3. You can pass additional hyperparameters such as model type, directly when running the experiment with:
   ``accelerate launch --config_file config/accelerator_config.yaml sft.py --model gpt2``