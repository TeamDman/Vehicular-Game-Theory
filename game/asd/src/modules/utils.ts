// inclusive rand
export function randI(min, max) {
    return Math.floor(Math.random() * (max - min + 1)) + min;
}

// inclusive rand int
export function rand(min, max) {
    const rtn = Math.random() * (max - min) + min;
    return toFixed(rtn);
}

export function toFixed(val) {
    return Math.round(val*100)/100;
}