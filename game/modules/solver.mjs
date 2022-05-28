import { knapsack } from "./knapsack.mjs";

export function best(
    board,
    costFunc,
    valueFunc,
    capacity
) {
    return knapsack(board, costFunc, valueFunc, capacity);
}

export function worst(
    board,
    costFunc,
    valueFunc,
    capacity
) {
    return knapsack(board, costFunc, x=>1/valueFunc(x), capacity);
}

export const strategies = {
    best,
    worst,
};