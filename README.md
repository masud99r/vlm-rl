Code for the paper (NAACL 2024): Natural Language-based State Representation in Deep Reinforcement Learning

## Overview
This codebase is designed for developing and evaluating reinforcement learning (RL) algorithms that leverage both image-based and text-based observations. It provides tools to process frozen lake environments, generate textual and visual representations, and apply Proximal Policy Optimization (PPO) for agent training and evaluation. This framework is especially suited for tasks requiring multimodal observation processing.

## Directory Structure

### RL Exp
- **`env_data_full`**: Contains datasets of environment states as images.
- **`ppo_image.py`**: Implements PPO training using image-based observations.
- **`ppo_text.py`**: Implements PPO training using text-based observations.

### VLM Processing
- **`frozen_lake`**: A directory containing frozen lake images representing different states.
- **`frozen_lake_blue`**: A variant of frozen lake images with a different color scheme.
- **`frozen_lake_blue_text_dict`**: Preprocessed text descriptions of frozen lake states in the blue variant.
- **`frozen_lake_blue_state_dict`**: Numerical representations (e.g., embeddings) of the blue frozen lake processed text states.
- **`frozen_lake_state_dict`**: Numerical representations (e.g., embeddings) of the standard frozen lake processed text states.
- **`frozen_lake_text_dict`**: Preprocessed text descriptions of standard frozen lake states.
- **`text_mdp.py`**: Scripts for generating text-based representations of environment states using pretrained models.

The following dependencies of [LLAVA](https://github.com/haotian-liu/LLaVA) and [CleanRL](https://github.com/vwxyzjn/cleanrl) are required to run the code.

## Key Components


### `ppo_image.py`
This script trains an RL agent using image-based observations.
- **Highlights**:
  - Uses `gym` for environment simulation.
  - Converts frozen lake environment states into image representations.
  - Code is based on [CleanRL](https://github.com/vwxyzjn/cleanrl).

### `ppo_text.py`
This script trains an RL agent using text-based observations.
- **Highlights**:
  - Utilizes precomputed text embeddings for environment states.
  - Uses `gym` for environment simulation.
  - Converts frozen lake environment states into image representations.
  - Code is based on [CleanRL](https://github.com/vwxyzjn/cleanrl).

### `text_mdp.py`
A preprocessing utility to generate textual and numerical representations of environment states.
- **Features**:
  - Leverages pretrained sentence transformers for text embeddings.
  - Saves outputs as dictionaries for further use in RL training.
  - Sample prompt: "Describe the observation"
