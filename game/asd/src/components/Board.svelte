<script lang="ts">
	import type { Action, Board, Player } from 'src/app';
	import { createEventDispatcher } from 'svelte';

	export let board: Board;
	export let attacking: Set<Action>;
	export let defending: Set<Action>;

	const dispatch = createEventDispatcher();

	$: componentCount = board.map((x) => x.comp).reduce((a, v) => Math.max(a, v + 1), 0);
	$: vulnCount = board.map((x) => x.vuln).reduce((a, v) => Math.max(a, v + 1), 0);

	function getAction(board: Board, i: number, j: number) {
		return board.find((x) => x.comp === i && x.vuln === j) ?? null;
	}
	function getDesc(action?: Action) {
		if (action === null) return '';
		const keys: (keyof Action)[] = ['attackCost', 'defendCost', 'attackProb', 'defendProb', 'risk', 'severity'];
		let desc = '';
		for (const key of keys) {
			desc += `${key}: ${(action as any)[key]}\n`;
		}
		return desc;
	}
	function toggle(action: Action, player: Player) {
		dispatch('toggle', { action, player });
	}
</script>

Board example

<table>
	<thead>
		<tr>
			<th />
			{#each Array(vulnCount) as _, i}
				<th>Vuln {i + 1}</th>
			{/each}
		</tr>
	</thead>
	<tbody>
		{#each Array(componentCount) as _, i}
			<tr>
				<td>Comp {i}</td>
				{#each Array(vulnCount) as _, j}
					{@const action = getAction(board, i, j)}
					{#if action}
						<td
							on:click={() => toggle(action, 'attacker')}
							on:contextmenu|preventDefault={() => toggle(action, 'defender')}
							class="noselect"
							class:attacking="{attacking.has(action)}"
							class:defending="{defending.has(action)}"
						>
							{getDesc(action)}</td
						>
					{:else}
						<td class="inactive" />
					{/if}
				{/each}
			</tr>
		{/each}
	</tbody>
</table>

<style>
	:root {
		--atk: rgba(240, 128, 128, 0.712);
		--def: #2c4fc4a9;
	}

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

	.inactive {
		background-color: gray;
	}

	.attacking:not(.defending) {
		background-color: var(--atk);
	}

	.defending:not(.attacking) {
		background-color: var(--def);
	}

	.attacking.defending {
		background: linear-gradient(to left bottom, var(--atk) 50%, var(--def) 50%);
	}

	.noselect {
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
