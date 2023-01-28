# Todo

## My own

- [ ] Rename "sub-game" and "platoon-game" to be "environments" instead with better names
- [ ] inline todo items

## Meeting notes from 2023-01-23

- Make sure to use same term everywhere
    - [x] CAV vs VANET vs Autonomous vehicles
- [x] Move RL to 3; Reflow to only introduce RL after subgame
- [ ] Add chapter to background for applications of game theory and RL in security problems with platoons and vehicles
- [ ] Add chapter to background for surveys of vehicle threats and defenses before [2.3 risk estimation]
- [ ] Add chapter in background after 2.4 for platooning - autonomous and not
- [x] Break out methodology into two distinct chapters
- [ ] Review platoon formalization to ensure matches latest

Prepare a response to question
- Why did you use PPO/RayRLLib instead of your own impl?

- intra, risk, game, vehicle
- Intra-vehicle security-conscious game formulation and solving
- Inter-vehicle security-conscious platoon formalization and operation

- framework to dynamically form platoon by evaluating risk in model free environments
- using game theory in the vehicle


If bored, jump ahead Ch 2. theory and Ch 6. results
Scenarios, results, metrics

## Meeting notes from 2023-01-16

- dynamic
- platoon
- formation
- reinforcement learning
- security risk
- protection
- prevention
Security risk based dynamic platoon formation using reinforcement learning
"security risk? define"
"end goal? why form platoon using RL and risk?"
key contribution: forming platoons using RL in a security-conscious manner
> learning the environment by interacting with it - model free
MDP for platoon -> security risk, membership, efficiency
*RL used to approximate a highly dynamic environment that we don't have a model for*
dataset/experiences obtained via exploration
"why RL? justify"
related works: differences and highlights
- existing platoon formation paper => less secure than ours
- when they finish reading related work, they understand how ours is contributative
- compare with other platoon formation patterns
- background chapter: on security risk assessment
- background chapter: game theory risk approximation/subgame
- impact of the work? platoon formation now more secure. What do I get as user? layman explanation of improvements

Q's:
- How can we form platoons in a secure way?
- How can we form platoons dynamically
- How to tell which vehicles are safe in an env we don't have a model for?

- Look at other master's theses
    - structure
    - titles
    - abstracts
    - introductions
    - from Queen's?
    - go back to Title after abstract, constant revision
    - length of each section

- Prioritize writing background first over introduction

1. Title
2. Abstract
3. Table of Contents
4. Introduction
5. Background, related works, lit review
6. Methods and experiments
7. Evaluation and results
8. Conclusion

ToDo:
1. Review existing theses
2. Write working title
3. working abstract
4. background

in parallel:
1. Implement DQN
2. Add DQN to graph
3. Send to talal

1.

1. Talal to get permission to attend defence

Timeline
- 1 month lead time to organize defense
- target: end of March ready to present
    - worst case: end of May