# Entropy-Regularized Token-Level Policy Optimization (ETPO)
This is the **official implementation** of ETPO (https://arxiv.org/abs/2402.06700), an entropy-augmented RL method tailored for optimizing LLMs at the token level, improving LLMs' sequential decision-making ability. This repo currently supports data science code generation tasks, broader tasks will be added soon.


## To Run:
1. cd etpo/scripts/
2. bash train_llm.sh

## Note:
1. see etpo/scripts/train_llm.sh for dataset selection
2. Or see etpo/scripts/train_llm_agent.py for algorithm and dataset-related hyper-parameters
3. see config.py in the root folder for more detailed hyper-parameters, there might be many redundant args, just ignore them ~
4. For agent API and training algorithm API, see agents/llama_agent.py and trainers/llm_trainer_etpo.py
5. For the entire rollout and training pipeline, see runner/shared/llm_agent_runner.py
6. log will be stored in etpo/scripts/results/ 

