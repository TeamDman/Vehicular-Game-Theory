import { valueFunc } from "./board.mjs";

export function rebuildTable(board) {
    const componentCount = board.map(x=>x.comp).reduce((a,v)=>Math.max(a,v+1),0);
    const vulnCount = board.map(x=>x.vuln).reduce((a,v)=>Math.max(a,v+1),0);
    console.log(componentCount, vulnCount);
    const table = document.querySelector("table");
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
            const entry = board.filter(x => `${x.comp},${x.vuln}` === `${i},${j}`)?.[0];
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
                    entry.attacking = !entry.attacking;
                    rebuildLabels(board);
                });

                value.addEventListener("contextmenu", e => {
                    value.classList.toggle("defend");
                    entry.defending = !entry.defending;
                    rebuildLabels(board);
                    e.preventDefault();
                });

                if (entry.attacking) value.classList.add("attack");
                if (entry.defending) value.classList.add("defend");
            }
            elem.appendChild(value);
        }
        tableBody.appendChild(elem);
    }
}

export function rebuildLabels(board) {
    const attackCost = document.querySelector("#attackcost");
    const defendCost = document.querySelector("#defendcost");
    attackCost.innerText = board.filter(x => x.attacking).reduce((a, b) => a += b.costAttack, 0);
    defendCost.innerText = board.filter(x => x.defending).reduce((a, b) => a += b.costDefend, 0);
    const total = document.querySelector("#total");
    total.innerText = valueFunc(board);
}

export function render(board) {
    rebuildTable(board);
    rebuildLabels(board);
}