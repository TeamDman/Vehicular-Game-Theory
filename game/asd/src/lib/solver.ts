import { knapsack } from "./knapsack";
import { generateBoard } from "./board";
import { toFixed } from "./utils";
import type { Action, Board, Player, State, Strategy } from "src/app";

export function valueFunc(state: Omit<State, "attackerCapacity" | "defenderCapacity">) {
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

export const none: Strategy = ( board, costFunc, probFunc, capacity ) => {
    return new Set<string>();
}


export const strategies: {
    [key:string]:Strategy;
} = {
    best,
    worst,
    none,
};

export function applyStrat(strat: Strategy, player: Player, board: Board, capacity: number) {
    if (player === "attacker") {
        return strat(board, x=>x.attackCost, x=>x.attackProb, capacity)
    } else if (player === "defender") {
        return strat(board, x=>x.defendCost, x=>x.defendProb, capacity)
    }
    throw new Error("bad player");
}

export function evaluateStrategies(
    rounds = 100,
    attackingCapacity = 10,
    defendingCapacity = 10,
) {
    const results: {
        [key:string] : {
            [key: string] : number[];
        };
    } = {};
    for (let i = 0; i < rounds; i++) {
        const board = generateBoard();
        for (const [defStratName, defStratFunc] of Object.entries(strategies)) {
            if (results?.[defStratName] === undefined) results[defStratName] = {};
            for (const [atkStratName, atkStratFunc] of Object.entries(strategies)) {
                if (results[defStratName]?.[atkStratName] === undefined) results[defStratName][atkStratName] = [];
                const entry = results[defStratName][atkStratName];
                const state = {
                    board,
                    attacking: applyStrat(atkStratFunc, "attacker", board, attackingCapacity),
                    defending: applyStrat(defStratFunc, "defender", board, defendingCapacity),
                }
                entry.push(valueFunc(state));
            }
        }
    }
    return results;
    // const rtn = [];
    // for (const [atkName, atkVal] of Object.entries(results)) {
    //     for (const [defName, defVal] of Object.entries(atkVal)) {
    //         rtn.push({
    //             attackStrat: atkName,
    //             defendStrat: defName,
    //             average: toFixed(defVal.reduce((acc, v) => acc + v, 0) / defVal.length),
    //         });
    //     }
    // }
    // return rtn;
}