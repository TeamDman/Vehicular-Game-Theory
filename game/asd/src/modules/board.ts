import { rand, randI } from "./utils";

export function getRisk(action: import("src/app.js").Action) {
    const rtn = action.severity ** 2 * action.probAttack * (1-action.probDefend) * (action.costDefend / action.costAttack);
    return Math.round(rtn*100)/100;
}

export function generateBoard(
    options = {
        numComponents: 5,
        numVulnerabilities: 4,
        weaknesses: [0,1,3,2,0]
    }
): Board {
    const rtn = [];
    for (let i=0; i<options.numComponents; i++) {
        for (let j=0; j<options.numVulnerabilities-options.weaknesses[i]; j++) {
            const entry = {
                key: `${i},${j}`,
                comp: i,
                vuln: j,
                attackCost: randI(1, 10),
                defendCost: randI(1, 10),
                attackProb: rand(0.5, 0.98),
                defendProb: rand(0.5, 0.98),
                severity: randI(1,5),
            };
            entry.risk = getRisk(entry);
            rtn.push(entry);
        }
    }
    return rtn;
}