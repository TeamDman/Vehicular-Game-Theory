import { knapsack } from "./knapsack.mjs";
import { generateBoard } from "./board.mjs";
import { toFixed } from "./utils.mjs";

export function valueFunc(state) {
    const rtn = state.board
        .filter(v => state.attacking.has(v.key))
        .reduce((acc, v) => {
            let inc = (v.severity ** 2) * v.probAttack;
            if (state.defending.has(v.key)) inc *= 1-v.probDefend;
            return acc + inc;
        }, 0);
    return Math.round(rtn*100)/100;
}

export function best(
    board,
    costFunc,
    probFunc,
    capacity
) {
    const rtn = knapsack(board, costFunc, x => x.severity, capacity);
    return rtn.subset.reduce((acc, v) => acc.add(v.key), new Set());
}

export function worst(
    board,
    costFunc,
    probFunc,
    capacity
) {
    const rtn = knapsack(board, costFunc, x => 1 / x.severity, capacity);
    return rtn.subset.reduce((acc, v) => acc.add(v.key), new Set());
}

export const strategies = {
    best,
    worst,
};

export function evalFor(strat, player, board, capacity) {
    if (player === "attacker") {
        return strat(board, x=>x.costAttack, x=>x.probAttack, capacity)
    } else if (player === "defender") {
        return strat(board, x=>x.costDefend, x=>x.probDefend, capacity)
    }
    throw new Error("bad player");
}

export function evaluateStrategies(
    attackingCapacity = 10,
    defendingCapacity = 10,
) {
    const results = {};
    for (let i = 0; i < 100; i++) {
        const board = generateBoard();
        for (const [atkStratName, atkStratFunc] of Object.entries(strategies)) {
            if (results?.[atkStratName] === undefined) results[atkStratName] = {};
            for (const [defStratName, defStratFunc] of Object.entries(strategies)) {
                if (results[atkStratName]?.[defStratName] === undefined) results[atkStratName][defStratName] = [];
                const entry = results[atkStratName][defStratName];
                const state = {
                    board,
                    attacking: evalFor(atkStratFunc, "attacker", board, attackingCapacity),
                    defending: evalFor(defStratFunc, "defender", board, defendingCapacity),
                }
                entry.push(valueFunc(state));
            }
        }
    }
    const rtn = [];
    for (const [atkName, atkVal] of Object.entries(results)) {
        for (const [defName, defVal] of Object.entries(atkVal)) {
            rtn.push({
                attackStrat: atkName,
                defendStrat: defName,
                average: toFixed(defVal.reduce((acc, v) => acc + v, 0) / defVal.length),
            });
        }
    }
    return rtn;
}