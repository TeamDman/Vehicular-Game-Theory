# Vehicular-Game-Theory

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/TeamDman/Vehicular-Game-Theory/blob/master/platoongame/training.ipynb)

New notebook:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/TeamDman/Vehicular-Game-Theory/blob/master/platoongame/rage.ipynb)

## SEE ALSO

https://github.com/primetang/pyflann
https://github.com/nikhil3456/Deep-Reinforcement-Learning-in-Large-Discrete-Action-Spaces
https://github.com/ghliu/pytorch-ddpg

## Environments

### Platoon-V0

- 10 vehicles
- Each vehicle starts as "out" of the platoon
- Action 0-10 toggles vehicle _i_ as "in" or "out". Action 0 is a no-op
- Reward is $-1 * count of vehicles outside of platoon$

### Platoon-V1

- 10 vehicles
- 2 of the vehicles are worth -10, the rest are worth 0
- Each vehicle starts as "out"
- Action 0-10 toggles vehicle, 0 is no-op
- Reward is $(-1 * count of vehicles outside the platoon) + (sum of vehicle values in the platoon)$

### Platoon-V2

- 10 vehicles
- Each vehicle starts as "out"
- Each vehicle has random value between -10 and 10, inclusive
- Action 0-10 toggles vehicle, 0 is no-op
- Reward is $(-1 * count of vehicles outside the platoon) + (sum of vehicle values in the platoon)$

### Platoon-V3

- 10 vehicles
- Each vehicle starts as "out"
- Each vehicle has random value between -5 and 0, inclusive
- Each vehicle starts with a random probability between 0 and 1, exclusive
- Action 0-10 toggles vehicle, 0 is no-op
- Each time step, one vehicle in the platoon is chosen. The probability of that vehicle becomes 1 with % chance according to the probability.
- Reward is $(-1 * count of vehicles outside the platoon) + sum(vehicle value * floor(vehicle probability) for each vehicles in the platoon)$

Evaluating the probability of a compromise is a one time thing, either it won't ever work for the vehicle or it always will.

## Notes to self

1509.02971.pdf CONTINUOUS CONTROL WITH DEEP REINFORCEMENT LEARNING  
p4. We create a copy of the actor and critic networks, Q′(s, a|θQ′) and μ′(s|θμ′) respectively, that are used for calculating the target values. The weights of these target networks are then updated by having them slowly track the learned networks: θ′ ← τ θ + (1 − τ )θ′ with τ  1. This means that the target values are constrained to change slowly, greatly improving the stability of learning. 


# Notes

RLLib slack response regarding my PPO not working

---

1. I personally would not use curiosity exploration until after I got a baseline version working first.
2. Does your observation include information about which row it is deciding an action for. You are not using framestacking or an rnn policy in your config. A pure feedforward policy could not learn to keep track of that information on its own.
3. You want to keep your rewards smaller preferably between 0 and 1. Also I have found that I get much better results it I return an instantaneous reward rather than cumulative on each step. So for example +/- for the square that was just turned on. Then at the end you can supply a terminal reward for the overall score. The reason is because targrt values will based on the return which will be a discounted sum of all the rewards in that episode. That adds a lot of extra counting of rewards and it makes it harder to distinguish good actions from bad ones (credit assignment).
4. Rllib's implementation of value clipping will Clio if the values loss is greater than 10. I would bet, given your rewards, that almost all of your experience steps are bring clipped. When a step is clipped, it will not be learned on. You can set the vf_clip to float("inf") to disable and see if that makes a difference. Whatcha or for Nan's during training if you do that though. (edited) 