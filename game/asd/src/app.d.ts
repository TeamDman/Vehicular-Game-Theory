/// <reference types="@sveltejs/kit" />

// See https://kit.svelte.dev/docs/types#app
// for information about these interfaces
declare namespace App {
	// interface Locals {}
	// interface Platform {}
	// interface Session {}
	// interface Stuff {}
}

export type Action = {
	key: string;
	comp: number;
	vuln: number;
	attackCost: number;
	defendCost: number;
	attackProb: number;
	defendProb: number;
	risk: number;
	severity: number;
}

export type Board = Action[]

export type Player = "attacker" | "defender"

export type Strategy = ( board: Board, capacity: any ) => Set<string>