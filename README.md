To Run:
1. cd mappo/scripts/
2. bash train_llm.sh

Note:
1. see mappo/scripts/train_llm.sh for dataset selection
2. Or see mappo/scripts/train_llm_agent.py for algorithm and dataset related hyper-parameters
3. see config.py in the root folder for more detailed configs, there might be many redundant args, just ignore it~
4. For agent API and training algorithm API, see agents/llama_agent.py and trainers/llm_trainer_etpo.py
5. For the entire rollout and training pipeline, see runner/shared/llm_agent_runner.py
6. log will be stored in mappo/scripts/results/ 

