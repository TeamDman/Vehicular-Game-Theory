<script lang="ts">
	import ValueBoard from '../components/ValueBoard.svelte';
	import MatchBoard from '../components/MatchBoard.svelte';
	import BoardComponent from '../components/Board.svelte';
	import Stats from '../components/Stats.svelte';
	import Importer from '../components/Importer.svelte';
	import Controls from '../components/Controls.svelte';
	import { generateBoard, type BoardOptions } from '../lib/board';
	import type { Action, Player, Board } from 'src/app';
	import { evaluateStrategies, strategies } from '$lib/solver';
	

	let attackerMaxCost = 10;
	let defenderMaxCost = 10;
	let attackerMinProb = 0.1;
	let attackerMaxProb = 0.98;
	let defenderMinProb = 0.1;
	let defenderMaxProb = 0.98;

	let attackerCapacity = 10;
	let defenderCapacity = 10;

	$: boardOptions = {
		numComponents: 5,
		numVulnerabilities: 4,
		weaknesses: [0,1,3,2,0],
		attackerMaxCost,
		defenderMaxCost,
		attackerMinProb,
		attackerMaxProb,
		defenderMinProb,
		defenderMaxProb,
	};

	$: board = generateBoard(boardOptions);
	let attacking: Set<Action> = new Set();
	let defending: Set<Action> = new Set();


	function toggle(event: CustomEvent<{ action: Action; player: Player }>) {
		const set = event.detail.player === 'attacker' ? attacking : defending;
		if (set.has(event.detail.action)) set.delete(event.detail.action);
		else set.add(event.detail.action);
		attacking = attacking;
		defending = defending;
	}

	function attack(event: CustomEvent<keyof typeof strategies>) {
		attacking = strategies[event.detail](board, 'attacker', attackerCapacity);
	}
	function defend(event: CustomEvent<keyof typeof strategies>) {
		defending = strategies[event.detail](board, 'defender', defenderCapacity);
	}

	
	function newBoard() {
		board = generateBoard(boardOptions);
	}

	function setBoard(b: Board) {
		board = b;
		attacking.clear();
		defending.clear();
	}

	let iterations = 100;
	$: results = evaluateStrategies(iterations, boardOptions, attackerCapacity, defenderCapacity);
</script>

<div style="display:flex">
	<div style="margin: 5px;">
		<BoardComponent {board} {attacking} {defending} on:toggle={toggle} />
		<div style="margin:5px;">
			<Importer {setBoard}/>
		</div>
	</div>
	<div style="margin:5px;">
		<div>
			<Stats {board} {attacking} {defending} />
		</div>
		<div style="margin-top:10px;">
			<Controls
				bind:attackerMaxCost
				bind:defenderMaxCost
				bind:attackerMinProb
				bind:attackerMaxProb
				bind:defenderMinProb
				bind:defenderMaxProb
				bind:attackerCapacity
				bind:defenderCapacity
				bind:iterations
				on:newboard={newBoard}
			/>
		</div>
	</div>
	<div>
		<div style="margin: 5px;">
			<ValueBoard {results} on:attack={attack} on:defend={defend} />
		</div>
		<div style="margin: 5px;">
			<MatchBoard {results} on:attack={attack} on:defend={defend} />
		</div>
	</div>
</div>
