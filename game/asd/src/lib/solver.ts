import { knapsack } from "./knapsack";
import { generateBoard, type BoardOptions } from "./board";
import { pickRandom, toFixed } from "./utils";
import type { Action, Board, Player, Strategy } from "src/app";
import { claim_space } from "svelte/internal";

export function valueFunc(board: Board, attacking: Set<Action>, defending: Set<Action>) {
    const rtn = Array.from(attacking)
        .reduce((acc, v) => {
            let inc = (v.severity ** 2) * v.attackProb;
            if (defending.has(v)) inc *= 1 - v.defendProb;
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
    return knapsack(board, getCostFunc(player), x => x.severity, capacity);
}

export const bestRisk: Strategy = (board, player, capacity) => {
    return knapsack(board, getCostFunc(player), x => x.risk, capacity);
}

export const bestRiskRandom: Strategy = (board, player, capacity) => {
    const rtn = new Set<Action>();
    const costFunc = getCostFunc(player);
    while (capacity > 0) {
        const candidates = board.filter(x => !rtn.has(x)).filter(x => costFunc(x) <= capacity).sort((a,b) => b.risk - a.risk).slice(0, 5);
        if (candidates.length === 0) return rtn;
        const elem = pickRandom(Array.from(candidates));
        capacity -= costFunc(elem);
        rtn.add(elem);
    }
    return rtn;
}

// not the most cost-effective for us, but the defender wants to prioritize the items the opponent wants anyways
export const bestGuess: Strategy = (board, player, capacity) => {
    const otherPlayer: Player = player === "attacker" ? "defender" : "attacker";
    const costFunc = getCostFunc(player);
    const theirGoal = bestRisk(board, otherPlayer, capacity); // assume same capacity
    const rtn = bestRisk(Array.from(theirGoal), player, capacity);
    capacity -= Array.from(rtn).map(costFunc).reduce((a,b) => a+b,0);
    Array.from(bestRisk(board.filter(x => !rtn.has(x)), player, capacity)).forEach(x => rtn.add(x)); // use up remaining capacity
    return rtn;
}

export const cheap: Strategy = (board, player, capacity) => {
    return knapsack(board, getCostFunc(player), x => getCostFunc(player)(x), capacity);
}

export const worstRisk: Strategy = (board, player, capacity) => {
    return knapsack(board, getCostFunc(player), x => 1/x.risk, capacity);
}

export const random: Strategy = (board, player, capacity) => {
    const rtn = new Set<Action>();
    while (capacity > 0) {
        const candidates = board.filter(x => !rtn.has(x)).filter(x => getCostFunc(player)(x) <= capacity);
        if (candidates.length === 0) return rtn;
        const entry = pickRandom(candidates);
        rtn.add(entry);
        capacity -= getCostFunc(player)(entry);
    }
    return rtn;
}

export const none: Strategy = (board, costFunc, capacity) => {
    return new Set<Action>();
}


export const strategies: {
    [key: string]: Strategy;
} = {
    bestRisk,
    bestRiskRandom,
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
    const results: {
        [key: string]: {
            [key: string]: {
                board: Board;
                attacking: Set<Action>;
                defending: Set<Action>;
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
                if (board.filter(x => state.attacking.has(x)).map(x=>x.attackCost).reduce((a,b) => a+b,0) > attackingCapacity) throw new Error("Exceeded atk cap");
                if (board.filter(x => state.defending.has(x)).map(x=>x.defendCost).reduce((a,b) => a+b,0) > defendingCapacity) throw new Error("Exceeded def cap");
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