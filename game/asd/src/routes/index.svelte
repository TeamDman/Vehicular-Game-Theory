<script lang="ts">
	import ValueBoard from '../components/ValueBoard.svelte';
	import MatchBoard from '../components/MatchBoard.svelte';
	import Board from '../components/Board.svelte';
	import Stats from '../components/Stats.svelte';
	import Controls from '../components/Controls.svelte';
	import { generateBoard } from '../lib/board';
	import type { Action, Player } from 'src/app';
	import { applyStrat, evaluateStrategies, strategies } from '$lib/solver';

	let attackerCapacity = 10;
	let defenderCapacity = 10;
	let board = generateBoard();
	let attacking = new Set<string>();
	let defending = new Set<string>();

	function newBoard() {
		board = generateBoard();
	}

	function toggle(event: CustomEvent<{ action: Action; player: Player }>) {
		const set = event.detail.player === 'attacker' ? attacking : defending;
		if (set.has(event.detail.action.key)) set.delete(event.detail.action.key);
		else set.add(event.detail.action.key);
		attacking = attacking;
		defending = defending;
	}

	function attack(event: CustomEvent<keyof typeof strategies>) {
		attacking = applyStrat(strategies[event.detail], board, attackerCapacity);
	}
	function defend(event: CustomEvent<keyof typeof strategies>) {
		defending = applyStrat(strategies[event.detail], board, defenderCapacity);
	}

	$: results = evaluateStrategies(100, attackerCapacity, defenderCapacity);
</script>

<div style="display:flex">
	<div style="margin: 5px;">
		<Board board={board} attacking={attacking} defending={defending} on:toggle={toggle} />
	</div>
	<div>
		<div>
			<Stats board={board} attacking={attacking} defending={defending} />
		</div>
		<div style="margin-top:10px;">
			<Controls bind:attackerCapacity={attackerCapacity} bind:defenderCapacity={defenderCapacity} on:newboard={newBoard} />
		</div>
	</div>
</div>

<div style="margin: 5px;">
	<ValueBoard {results} on:attack={attack} on:defend={defend} />
</div>
<div style="margin: 5px;">
	<MatchBoard {results} on:attack={attack} on:defend={defend} />
</div>
