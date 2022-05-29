import { knapsack } from "./knapsack";
import { generateBoard } from "./board";
import { toFixed } from "./utils";
import type { Action, Board, Player, State, Strategy } from "src/app";

export function valueFunc(state: State) {
    const rtn = state.board
        .filter(v => state.attacking.has(v.key))
        .reduce((acc, v) => {
            let inc = (v.severity ** 2) * v.attackProb;
            if (state.defending.has(v.key)) inc *= 1-v.defendProb;
            return acc + inc;
        }, 0);
    return Math.round(rtn*100)/100;
}

export const best: Strategy = (
    board,
    costFunc,
    probFunc,
    capacity
) => {
    const rtn = knapsack(board, costFunc, (x: Action) => x.severity, capacity).subset;
    return rtn.reduce((acc, v) => acc.add(v.key), new Set<string>());
}

export const worst: Strategy = (
    board,
    costFunc,
    probFunc,
    capacity
) => {
    const rtn = knapsack(board, costFunc, (x: Action) => 1 / x.severity, capacity).subset;
    return rtn.reduce((acc, v) => acc.add(v.key), new Set<string>());
}



export const strategies: {
    [key:string]:Strategy;
} = {
    best,
    worst,
};

export function evalFor(strat: Strategy, player: Player, board: Board, capacity: number) {
    if (player === "attacker") {
        return strat(board, x=>x.attackCost, x=>x.attackProb, capacity)
    } else if (player === "defender") {
        return strat(board, x=>x.defendCost, x=>x.defendProb, capacity)
    }
    throw new Error("bad player");
}

export function evaluateStrategies(
    attackingCapacity = 10,
    defendingCapacity = 10,
) {
    const results: {
        [key:string] : {
            [key: string] : number[];
        };
    } = {};
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