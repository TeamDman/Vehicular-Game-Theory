<script lang="ts">
	import type { Action, Player, State } from 'src/app';
	import { createEventDispatcher } from 'svelte';
	import { evaluateStrategies, strategies, valueFunc } from '$lib/solver';
	import { pickHex, toFixed } from '$lib/utils';

	const dispatch = createEventDispatcher();

	const strats = Object.keys(strategies);
	$: results = evaluateStrategies(100);

	function calc(states: Pick<State, "attacking" | "defending" | "board">[]) {
		let matches = 0;
		let total = 0;
		for (const state of states) {
			total+=state.attacking.size;
			matches += Array.from(state.attacking).filter(x => state.defending.has(x)).length;
		}
		if (matches === 0) return Array.from(states).every(x => x.attacking.size === 0) ? 100 : 0;
		return toFixed(matches / total * 100);
	}
</script>

Match percentage

<table>
	<thead>
		<tr>
			<th rowspan="2" colspan="2" />
			<th colspan={Object.keys(strategies).length}>Attacker</th></tr
		>
		<tr>
			{#each strats as strat, i}
				<th class="strat" on:click={() => dispatch('attack', strat)}>{strat}</th>
			{/each}
		</tr>
	</thead>
	<tbody>
		<th rowspan={Object.keys(strategies).length+1}>Defender</th>
		{#each strats as defStrat, i}
			<tr>
				<th class="strat" on:click={() => dispatch('defend', defStrat)}>{defStrat}</th>
				{#each strats as atkStrat, j}
					{@const result = calc(results[defStrat][atkStrat])}
					{@const rgb = pickHex([70,255,70], [240,70,70], result/100)}
					<td style="background-color:rgb({rgb[0]},{rgb[1]},{rgb[2]})">{result}%</td>
				{/each}
			</tr>
		{/each}
	</tbody>
</table>

<style>
	th,
	td {
		padding: 5px;
		white-space: pre-line;
		border: 2px solid;
		border-top-color: black;
		border-left-color: black;
		border-right-color: #777;
		border-bottom-color: #777;
	}
	table {
		border: 1px solid black;
	}

	.strat {
		cursor: pointer;
		-webkit-touch-callout: none;
		/* iOS Safari */
		-webkit-user-select: none;
		/* Safari */
		-khtml-user-select: none;
		/* Konqueror HTML */
		-moz-user-select: none;
		/* Old versions of Firefox */
		-ms-user-select: none;
		/* Internet Explorer/Edge */
		user-select: none;
		/* Non-prefixed version, currently
                                    supported by Chrome, Edge, Opera and Firefox */
	}
</style>
