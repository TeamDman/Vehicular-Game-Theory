# Todo

## My own

### Tasks

- [ ] inline todo items
- [ ] install lovelytensors https://github.com/xl0/lovely-tensors
- [ ] perform analysis with solo-game derived vehicles

### Questions

- [x] Why did we use PPO and DQN instead of other algos?
- [ ] Why did we use Ray RLLib instead of other libraries or our own impl?

## Overleaf 2023-02-19

- [ ] Inline chapter 2 comments
- [ ] Section 2.5 add more security stuff
    - Roy, S., Ellis, C., Shiva, S., Dasgupta, D., Shandilya, V., & Wu, Q. (2010, January). A survey of game theory as applied to network security. In 2010 43rd Hawaii International Conference on System Sciences (pp. 1-10). IEEE.
    

## Meeting notes 2023-02-13

- [x] Finish ch3 by weekend/sunday night
- [x] Finish ch4 by weekend/sunday night

- Consider how to boost the security aspect of Ch 4 

## Meeting notes 2023-02-06

- [x] Move platoon background to ch 2
- [x] Finish glossary adoption
- [x] fix bibtex warnings
- [x] Minimax objective section - see inline notes
- [ ] RL diagram
- [ ] Platoon objective section - see inline notes
- [ ] DQN explanation
    - [ ] Bellman explanation
- [ ] PPO explanation


## Meeting notes from 2023-01-30

- [x] Renaming of sub-game and platoon-game single-vehicle and platoon maybe? Inter- and Intra-platoon 
- [x] 3.1 Add literature section: game theory for security, including motivation
- [x] 4.1 Reinforcement learning for security literature including motivation
- [x] 2. Add section: game theory and security in vehicles
    - [x] Reference: Aawista Chaudhry, A Framework for Modeling Advanced Persistent Threats in Intelligent Transportation Systems, MSc, 2021.
- [x] Add technical background on platoon formation
    - [x] formation
    - [x] leader election
    - [ ] consensus mechanisms
- [x] Add overview tables for symbols used in Ch 3
- [ ] Bonus: add some content to results section

- Keep in mind: replication details. Anyone reading the paper should be able to reproduce the work.
    - Add github url in footnotes
- **Clean up repo**

- "Explain everything that you have done"
    - Why did you choose certain parameters?
    - Why did you choose RLLib? Other libs tried?

## Meeting notes from 2023-01-23

- Make sure to use same term everywhere
    - [x] CAV vs VANET vs Autonomous vehicles
- [x] Move RL to 3; Reflow to only introduce RL after subgame
- [x] Add chapter to background for applications of game theory and RL in security problems with platoons and vehicles
- [x] Add chapter to background for surveys of vehicle threats and defenses before [2.3 risk estimation]
- [x] Add chapter in background after 2.4 for platooning - autonomous and not
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