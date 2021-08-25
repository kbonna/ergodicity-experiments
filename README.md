# Experiments in Ergodicity 

This repository contains code for simulating decision-making experiments inspired by and based on [this work](https://arxiv.org/abs/1906.04652). 

The overarching aim of this part of the project is to design an experiment that would **give evidence in favor of one of the competing models of decision making under uncertainty**. Two currently prevailing models of risky decisions in economics are expected utility model (EUT) and prospect theory (PT). However, when different wealth dynamics are considered, ergodic considerations start to play important role in understanding time-average growth of agent's wealth. Ergodic considerations led to formulation of [competing time optimality model (TO)](https://arxiv.org/abs/1801.03680) based on the assumption that agents try to maximize the rate of change of their wealth over time. Simulations try answer the question – What is the optimal experimental design that maximizes disagreement between EUT, PT and TO models?  

## Summer school talk

You can hear more about the project and simulations watching my Summer school [final presentation](https://www.youtube.com/watch?v=ohOAPWXhrZg).

## How to reproduce results?

1. Make sure you have [Conda](https://conda.io/projects/conda/en/latest/user-guide/install/index.html) installed on your local machine.
2. Recreate conda environment using `environment.yml` file (for more help look [here](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file)):
```
conda env create -f environment.yml
```
3. Run your environment, open one of the notebooks and play around with the code.
