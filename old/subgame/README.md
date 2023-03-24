# Subgame

- Attacker and Defender play at the same time

## Actions

Each action has values for:

- Component ordinal being attacked
- Vulnerability ordinal being used
- Attacker cost $c_a$
- Attacker success probability $p_a$
- Defender cost $c_d$
- Defender success probability $p_d$
- Severity $s$

Derived values include:

- Risk $$r = s ^ 2 * p_a * (1-p_d) * \frac{c_d}{c_a}$$

## Scoring

Boards are scored using the sum of $s^2 * p_a$ for each attack chosen.  
If the value is also selected by the defender, the value $s^2 * p_a * (1-p_d)$ is used in the summation instead.