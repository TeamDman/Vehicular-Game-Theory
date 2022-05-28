import { valueFunc, strategies } from "./solver.mjs";

function toggleMembership(set, action) {
    if (set.has(action.key)) set.delete(action.key);
    else set.add(action.key);
}

export function rebuildBoardTable(state) {
    const componentCount = state.board.map(x=>x.comp).reduce((a,v)=>Math.max(a,v+1),0);
    const vulnCount = state.board.map(x=>x.vuln).reduce((a,v)=>Math.max(a,v+1),0);
    const table = document.querySelector("#board");
    const tableHead = table.querySelector("thead tr");
    const tableBody = table.querySelector("tbody");
    tableHead.innerHTML = "";
    tableBody.innerHTML = "";

    tableHead.appendChild(document.createElement("th"));
    for (var i = 1; i <= vulnCount; i++) {
        const elem = document.createElement("th");
        elem.innerText = "Vuln " + i;
        tableHead.appendChild(elem);
    }

    for (var i = 0; i < componentCount; i++) {
        const elem = document.createElement("tr");
        const label = document.createElement("td");
        label.innerText = "Comp " + i;
        elem.appendChild(label);
        for (var j = 0; j < vulnCount; j++) {
            const entry = state.board.filter(x => `${x.comp},${x.vuln}` === `${i},${j}`)?.[0];
            const value = document.createElement("td");
            value.classList.add("noselect");
            if (entry === undefined) {
                value.classList.add("invalid");
            } else {
                const keys = ["costAttack", "costDefend", "probAttack", "probDefend", "severity", "risk"];
                let desc = "";
                for (const key of keys) {
                    desc += `${key}: ${entry[key]}\n`;
                }
                
                value.innerText = desc.trimEnd();
                value.addEventListener("click", e => {
                    value.classList.toggle("attack");
                    toggleMembership(state.attacking, entry);
                    rebuildBoardLabels(state);
                });

                value.addEventListener("contextmenu", e => {
                    value.classList.toggle("defend");
                    toggleMembership(state.defending, entry);
                    rebuildBoardLabels(state);
                    e.preventDefault();
                });

                if (state.attacking.has(entry.key)) value.classList.add("attack");
                if (state.defending.has(entry.key)) value.classList.add("defend");
            }
            elem.appendChild(value);
        }
        tableBody.appendChild(elem);
    }
}

export function rebuildBoardLabels(state) {
    const attackCost = document.querySelector("#attackcost");
    const defendCost = document.querySelector("#defendcost");
    attackCost.innerText = state.board.filter(x => state.attacking.has(x.key)).reduce((a, b) => a += b.costAttack, 0);
    defendCost.innerText = state.board.filter(x => state.defending.has(x.key)).reduce((a, b) => a += b.costDefend, 0);
    const total = document.querySelector("#total");
    total.innerText = valueFunc(state);
}

export function renderBoard(state) {
    rebuildBoardTable(state);
    rebuildBoardLabels(state);
}

export function renderPickers() {
    const atk = document.querySelector("#pickatk");
    const def = document.querySelector("#pickdef");
    for(const v of Object.keys(strategies)) {
        const elem = document.createElement("option");
        elem.innerText = v;
        atk.appendChild(elem);
        def.appendChild(elem.cloneNode(true));
    }
}

export function rebuildComparisonTable() {

}


export function renderAll(state) {
    renderBoard(state);
    renderPickers();
}