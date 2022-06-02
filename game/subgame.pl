component(1).
component(2).
component(3).
component(4).
component(5).
vuln(1).
vuln(2).
vuln(3).
vuln(4).
weakness(component(1),4).
weakness(component(2),3).
weakness(component(3),1).
weakness(component(4),2).
weakness(component(5),4).
player(attacker).
player(defender).

attack(ComponentOrd,VulnOrd) :-
    component(ComponentOrd),
    vuln(VulnOrd),
    weakness(component(ComponentOrd), WeaknessOrd),
    VulnOrd =< WeaknessOrd.

severity(Attack,Sev) :-
    component(C),
    vuln(V),
    Attack = attack(C,V),
    random(1,5,Sev).

cost(Attack,Cost) :-
    component(C),
    vuln(V),
    Attack = attack(C,V),
    random(1,10,Cost).

cost(Player, Action, Cost) :-
    Player = player(attacker),
    Cost is Action.costAttack;
    Player = player(defender),
    Cost is Action.costDefend.

prob(Attack,Prob) :-
    component(C),
    vuln(V),
    Attack = attack(C,V),
    random(0.5, 1.0, Prob).

action(Attack, Action) :-
    % generate a random action
    component(C),
    vuln(V),
    Attack = attack(C,V),
    severity(Attack, S),
    cost(Attack, Ea),
    cost(Attack, Ed),
    prob(Attack, Pa),
    prob(Attack, Pd),
    Action = action{attack:Attack, severity:S, costAttack:Ea, costDefend:Ed, probabilityAttack:Pa, probabilityDefend:Pd},
    !.

attacks(A) :-
    bagof(attack(C,V), attack(C,V), A).

actions(Actions) :-
    attacks(Attacks),
    maplist(action, Attacks, Actions).

risk(Severity, PAttack, PDefend, AttackCost, DefendCost, Risk) :-
    Risk is (Severity ** 2) * PAttack * (1 - PDefend) * (DefendCost / AttackCost).
risk(Action, Risk) :-
    risk(Action.severity, Action.probabilityAttack, Action.probabilityDefend, Action.costAttack, Action.costDefend, Risk).

attackerPriority(Action, Priority) :-
    risk(Action, R),
    Priority is R / Action.costAttack.


defenderPriority(Action, Priority) :-
% in theory, this would be equal to attacker priority, since the defender would want to defend against all the attacks the attacker chooses
% in practice, just protect the "best value" items.
    risk(Action, R),
    Priority is R / Action.costDefend.

priority(Player, Action, Prio) :-
    (
        Player = player(attacker),
        attackerPriority(Action, P);
        Player = player(defender),
        defenderPriority(Action, P)
    ),
    Prio = P.

canAfford(Player, CostAvailable, Action) :-
    Player = player(attacker),
    Action.costAttack =< CostAvailable;
    Player = player(defender),
    Action.costDefend =< CostAvailable.

sortedByPriority(Actions, Player, Result) :-
    map_list_to_pairs(priority(Player), Actions, Pairs),
    keysort(Pairs, Asc),
    reverse(Asc, Desc),
    pairs_values(Desc, Result).

bestActionsNaive(Actions, Player, CostAvailable, [ THead | Rest]) :-
    include(canAfford(Player, CostAvailable), Actions, Affordable),
    sortedByPriority(Affordable, Player, [THead | _]),
    (
        Player = player(attacker),
        CostRemaining = CostAvailable - THead.costAttack;
        Player = player(defender),
        CostRemaining = CostAvailable - THead.costDefend
    ),
    (
        exclude(=(THead), Actions, Remaining),
        bestActionsNaive(Remaining, Player, CostRemaining, Rest);
        Rest=[]
    ),
    !.


% https://newtocode.wordpress.com/2013/11/23/knapsack-problem-in-prolog/
subseq([],[]).
subseq([Item | RestX], [Item | RestY]) :-
  subseq(RestX,RestY).
subseq([_ | Rest], X) :-
  subseq(Rest, X).

legal(Actions, Taken, CostFunc, CostAvailable) :-
    subseq(Actions, Taken),
    maplist(CostFunc, Taken, Costs),
    sum_list(Costs, TotalCost),
    TotalCost =< CostAvailable.

allLegalTurns(Actions, CostFunc, CostAvailable, LegalTurns) :-
    findall(Taken, legal(Actions, Taken, CostFunc, CostAvailable), LegalTurns).

scoreTurn(Turn, Score) :-
    maplist(risk, Turn, Risks),
    sum_list(Risks, Score).

bestTurn(Actions, CostFunc, CostAvailable, Result) :-
    allLegalTurns(Actions, CostFunc, CostAvailable, AllTurns),
    map_list_to_pairs(scoreTurn, AllTurns, TurnScores),
    max_member(_-Result, TurnScores).


testAttack :- 
    actions(A), write("Actions"), nl, write(A), nl, nl,
    bestActionsNaive(A, player(attacker), 10, R), write("Naive best attacker actions"), nl, write(R), nl, nl,
    maplist(risk, R, Z), write("Naive best attacker action risks"), nl, write(Z), sum_list(Z, Sum), write(" sum="), write(Sum), nl, nl,
    bestTurn(A, cost(player(attacker)), 10, T), write("Best attacker turn"), nl, write(T), nl, nl,
    maplist(risk, T, ZB), write("Best attacker action risks"), nl, write(ZB), sum_list(ZB, SumB), write(" sum="), write(SumB)
    . 
 
:- use_module(library(http/json)).
jsonTest :- 
    actions(A),
    bestActionsNaive(A, player(attacker), 10, NaiveActionsAttacker),
    bestTurn(A, cost(player(attacker)), 10, BestActionsAttacker),
    bestActionsNaive(A, player(defender), 10, NaiveActionsDefender),
    bestTurn(A, cost(player(defender)), 10, BestActionsDefender),

    open("./game.json", write, GameStream),
    json_write(GameStream, A, [serialize_unknown(true)]),
    close(GameStream),
    
    open("./naiveattacker.json", write, NaiveStreamAttacker),
    json_write(NaiveStreamAttacker, NaiveActionsAttacker, [serialize_unknown(true)]),
    close(NaiveStreamAttacker),
    open("./naivedefender.json", write, NaiveStreamDefender),
    json_write(NaiveStreamDefender, NaiveActionsDefender, [serialize_unknown(true)]),
    close(NaiveStreamDefender),
    open("./bestattacker.json", write, BestStreamAttacker),
    json_write(BestStreamAttacker, BestActionsAttacker, [serialize_unknown(true)]),
    close(BestStreamAttacker),
    open("./bestdefender.json", write, BestStreamDefender),
    json_write(BestStreamDefender, BestActionsDefender, [serialize_unknown(true)]),
    close(BestStreamDefender). 

%% todo: naive based purely on descending risk instead of priority