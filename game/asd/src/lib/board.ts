import type { Action, Board } from "src/app.js";
import { rand, randI } from "./utils";

export function getRisk(action: import("src/app.js").Action) {
    const rtn = action.severity ** 2 * action.attackProb * (1-action.defendProb) * (action.defendCost / action.attackCost);
    return Math.round(rtn*100)/100;
}

export type BoardOptions = {
    numComponents: number;
    numVulnerabilities: number;
    weaknesses: number[];
    attackerMaxCost: number;
    defenderMaxCost: number;
    attackerMinProb: number;
    attackerMaxProb: number;
    defenderMinProb: number;
    defenderMaxProb: number;
};

export function generateBoard( options: BoardOptions ): Board {
    const rtn = [];
    for (let i=0; i<options.numComponents; i++) {
        for (let j=0; j<options.numVulnerabilities-options.weaknesses[i]; j++) {
            const entry: Action = {
                key: `${i},${j}`,
                comp: i,
                vuln: j,
                attackCost: randI(1, options.attackerMaxCost),
                defendCost: randI(1, options.defenderMaxCost),
                attackProb: rand(options.attackerMinProb, options.attackerMaxProb),
                defendProb: rand(options.defenderMinProb, options.defenderMaxProb),
                severity: randI(1,5),
                risk: -1,
            };
            entry.risk = getRisk(entry);
            rtn.push(entry);
        }
    }
    return rtn;
}