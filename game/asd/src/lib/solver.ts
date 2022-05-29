import { knapsack } from "./knapsack";
import { generateBoard } from "./board";
import { toFixed } from "./utils";
import type { Board, Strategy } from "src/app";

export function valueFunc(board: Board, attacking: Set<string>, defending: Set<string>) {
    const rtn = board
        .filter(v => attacking.has(v.key))
        .reduce((acc, v) => {
            let inc = (v.severity ** 2) * v.attackProb;
            if (defending.has(v.key)) inc *= 1 - v.defendProb;
            return acc + inc;
        }, 0);
    return Math.round(rtn * 100) / 100;
}

export const attackerBest: Strategy = (board, capacity) => {
    const rtn = knapsack(board, x => x.attackCost, x => x.severity, capacity).subset;
    return rtn.reduce((acc, v) => acc.add(v.key), new Set<string>());
}

export const defenderBest: Strategy = (board, capacity) => {
    const rtn = knapsack(board, x => x.defendCost, x => x.severity, capacity).subset;
    return rtn.reduce((acc, v) => acc.add(v.key), new Set<string>());
}

export const attackerWorst: Strategy = (board, capacity) => {
    const rtn = knapsack(board, x => x.attackCost, x => 1 / x.severity, capacity).subset;
    return rtn.reduce((acc, v) => acc.add(v.key), new Set<string>());
}

export const defenderWorst: Strategy = (board, capacity) => {
    const rtn = knapsack(board, x => x.defendCost, x => 1 / x.severity, capacity).subset;
    return rtn.reduce((acc, v) => acc.add(v.key), new Set<string>());
}

export const none: Strategy = (board, capacity) => {
    return new Set<string>();
}


export const strategies: {
    [key: string]: Strategy;
} = {
    attackerBest,
    defenderBest,
    attackerWorst,
    defenderWorst,
    none,
};

export function applyStrat(strat: Strategy, board: Board, capacity: number) {
    return strat(board, capacity)
}

export function evaluateStrategies(
    rounds = 100,
    attackingCapacity = 10,
    defendingCapacity = 10,
) {
    console.log(`Evaluating ${rounds} rounds`);
    const results: {
        [key: string]: {
            [key: string]: {
                board: Board;
                attacking: Set<string>;
                defending: Set<string>;
            }[];
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
                    attacking: applyStrat(atkStratFunc, board, attackingCapacity),
                    defending: applyStrat(defStratFunc, board, defendingCapacity),
                }
                entry.push(state);
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