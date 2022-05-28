// inclusive rand
function randI(min, max) {
    return Math.floor(Math.random() * (max - min + 1)) + min;
}

// inclusive rand int
function rand(min, max) {
    const rtn = Math.random() * (max - min) + min;
    return Math.round(rtn*100)/100;
}

export function getRisk(action) {
    const rtn = action.severity ** 2 * action.probAttack * (1-action.probDefend) * (action.costDefend / action.costAttack);
    return Math.round(rtn*100)/100;
}

export function valueFunc(board) {
    const rtn = board
        .filter(x => x.attacking)
        .reduce((acc, v) => {
            let inc = (v.severity ** 2) * v.probAttack;
            if (v.defending) inc *= 1-v.probDefend;
            return acc + inc;
        }, 0);
    return Math.round(rtn*100)/100;
}

export function generateBoard(
    options = {
        numComponents: 5,
        numVulnerabilities: 4,
        weaknesses: [0,1,3,2,0]
    }
) {
    const rtn = [];
    for (let i=0; i<options.numComponents; i++) {
        for (let j=0; j<options.numVulnerabilities-options.weaknesses[i]; j++) {
            const entry = {
                comp: i,
                vuln: j,
                costAttack: randI(1, 10),
                costDefend: randI(1, 10),
                probAttack: rand(0.5, 0.98),
                probDefend: rand(0.5, 0.98),
                severity: randI(1,5),
                attacking: false,
                defending: false,
            };
            entry.risk = getRisk(entry);
            rtn.push(entry);
        }
    }
    return rtn;
}