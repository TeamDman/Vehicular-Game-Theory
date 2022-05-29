import { knapsack } from "./knapsack";
import { generateBoard, type BoardOptions } from "./board";
import { toFixed } from "./utils";
import type { Action, Board, Player, Strategy } from "src/app";
import { claim_space } from "svelte/internal";

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

function getCostFunc(player: Player) {
    return player === "attacker" ? (x: Action) => x.attackCost : (x: Action) => x.defendCost;
}
function getProbFunc(player: Player) {
    return player === "attacker" ? (x: Action) => x.attackProb : (x: Action) => x.defendProb;
}

export const bestSeverity: Strategy = (board, player, capacity) => {
    const rtn = knapsack(board, getCostFunc(player), x => x.severity, capacity).subset;
    return rtn.reduce((acc, v) => acc.add(v.key), new Set<string>());
}

export const bestRisk: Strategy = (board, player, capacity) => {
    const rtn = knapsack(board, getCostFunc(player), x => x.risk, capacity).subset;
    return rtn.reduce((acc, v) => acc.add(v.key), new Set<string>());
}

// not the most cost-effective for us, but the defender wants to prioritize the items the opponent wants anyways
export const bestGuess: Strategy = (board, player, capacity) => {
    const otherPlayer: Player = player === "attacker" ? "defender" : "attacker";
    const costFunc = getCostFunc(player);
    const theirGoal = bestRisk(board, otherPlayer, capacity); // assume same capacity
    const rtn = bestRisk(board.filter(x => theirGoal.has(x.key)), player, capacity);
    for (const x of board) { // try and fill any remaining space we have
        if (rtn.has(x.key)) continue;
        if (costFunc(x) <= capacity) {
            rtn.add(x.key);
            capacity -= costFunc(x);
        }
    }
    return rtn;

}

export const cheap: Strategy = (board, player, capacity) => {
    const rtn = knapsack(board, getCostFunc(player), x => getCostFunc(player)(x), capacity).subset;
    return rtn.reduce((acc, v) => acc.add(v.key), new Set<string>());
}

export const worstRisk: Strategy = (board, player, capacity) => {
    const rtn = knapsack(board, getCostFunc(player), x => 1/x.risk, capacity).subset;
    return rtn.reduce((acc, v) => acc.add(v.key), new Set<string>());
}

export const random: Strategy = (board, player, capacity) => {
    const rtn = new Set<string>();
    while (capacity > 0) {
        const candidates = board.filter(x => !rtn.has(x.key)).filter(x => getCostFunc(player)(x) <= capacity);
        if (candidates.length === 0) return rtn;
        const choice = Math.floor(Math.random() * candidates.length);
        const entry = candidates[choice];
        rtn.add(entry.key);
        capacity -= getCostFunc(player)(entry);
    }
    return rtn;
}

export const none: Strategy = (board, costFunc, capacity) => {
    return new Set<string>();
}


export const strategies: {
    [key: string]: Strategy;
} = {
    bestRisk,
    bestGuess,
    bestSeverity,
    cheap,
    random,
    worstRisk,
    none,
};

export function evaluateStrategies(
    rounds: number,
    boardOptions: BoardOptions,
    attackingCapacity: number,
    defendingCapacity: number,
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
        const board = generateBoard(boardOptions);
        for (const [defStratName, defStratFunc] of Object.entries(strategies)) {
            if (results?.[defStratName] === undefined) results[defStratName] = {};
            for (const [atkStratName, atkStratFunc] of Object.entries(strategies)) {
                if (results[defStratName]?.[atkStratName] === undefined) results[defStratName][atkStratName] = [];
                const entry = results[defStratName][atkStratName];
                const state = {
                    board,
                    attacking: atkStratFunc(board, "attacker", attackingCapacity),
                    defending: defStratFunc(board, "defender", defendingCapacity),
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