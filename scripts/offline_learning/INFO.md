## Online RL Experiment Setup

The ``run_offline_rl.py`` file loads a pre-trained model or a checkpoint and a offline dataset of
transition tuples $(s, a, r, s')$. This dataset is used to run offline RL.
During the training, we let the model interact with the AWS environment and evaluate its performance.
The folder ``logs`` is created where the trained checkpoints are saved. Furthermore, we store the history of
interactions between the agent and the environment in the logs.

### Running the experiments

1. Without accelerate: ``python run_offline_rl.py``
2. With accelerate: ``accelerate launch --config_file config/accelerator_config.yaml run_offline_rl.py``

### Key Considerations

1. When running multiple seeds on the same node/compute instance, make sure that the port number for the AWS environment
   is unique per run. This is crucial to avoid overlap in the environments from different runs. The seed is unique per
   run and we specify a unique port number for the moto server for each run. When we execute another run of the
   algorithm in parallel we have to make sure that the port where moto is deployed is empty. We can pass the port number
   to the python script as an argument or change it in the ``config/archer/aws.yaml`` file.
2. You can change the hyperparameters during training, e.g., learning rate, from ``config/archer/aws.yaml`` and
   ``config/archer/default.yaml``. ``config/archer/aws.yaml`` overrides some hyperparameters from
   ``config/archer/default.yaml``. The above example changes the hyperparameters for the Archer algorithm. You can
   change the hyperparameters for Filtered-SFT similarly.
3. You can pass additional hyperparameters such as algorithm name, directly when running the experiment with:
   ``accelerate launch --config_file config/accelerator_config.yaml run_online_rl.py --alg_name archer``