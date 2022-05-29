// inclusive rand
export function randI(min:number, max:number) {
    return Math.floor(Math.random() * (max - min + 1)) + min;
}

// inclusive rand int
export function rand(min:number, max:number) {
    const rtn = Math.random() * (max - min) + min;
    return toFixed(rtn);
}

export function toFixed(val:number) {
    return Math.round(val*100)/100;
}

export function pickHex(color1: [number, number, number], color2: [number, number, number], weight: number) {
    var w1 = weight;
    var w2 = 1 - w1;
    var rgb = [Math.round(color1[0] * w1 + color2[0] * w2),
        Math.round(color1[1] * w1 + color2[1] * w2),
        Math.round(color1[2] * w1 + color2[2] * w2)];
    return rgb;
}