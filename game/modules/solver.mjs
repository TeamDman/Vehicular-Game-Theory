import { knapsack } from "./knapsack.mjs";
import { generateBoard } from "./board.mjs";

export function best(
    board,
    costFunc,
    probFunc,
    capacity
) {
    return knapsack(board, costFunc, x => x.severity, capacity).sublist;
}

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


export function worst(
    board,
    costFunc,
    probFunc,
    capacity
) {
    return knapsack(board, costFunc, x => 1 / x.severity, capacity).sublist;
}

export const strategies = {
    best,
    worst,
};


export function evaluateStrategies() {
    const results = {};
    for (let i = 0; i < 100; i++) {
        const board = generateBoard();
        for (const [atkStratName, atkStratFunc] of Object.entries(strategies)) {
            if (results?.[atkStratName] === undefined) results[atkStratName] = {};
            for (const [defStratName, defStratFunc] of Object.entries(strategies)) {
                if (results[atkStratName]?.[defStratName] === undefined) results[atkStratName][defStratName] = {
                    values: []
                };
                const entry = results[atkStratName][defStratName];
                
                entry.values.push()
            }
        }
    }
    return results;
}