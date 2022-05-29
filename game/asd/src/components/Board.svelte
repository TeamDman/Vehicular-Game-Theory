<script lang="ts">
	import type { Action, Player, State } from 'src/app';
	import { createEventDispatcher } from 'svelte';

	export let state: State;
	const dispatch = createEventDispatcher();

	const componentCount = state.board.map((x) => x.comp).reduce((a, v) => Math.max(a, v + 1), 0);
	const vulnCount = state.board.map((x) => x.vuln).reduce((a, v) => Math.max(a, v + 1), 0);

	function getAction(i: number, j: number) {
		return state.board.find((x) => x.comp === i && x.vuln === j) ?? null;
	}
	function getDesc(action?: Action) {
		if (action === null) return '';
		const keys = ['attackCost', 'defendCost', 'attackProb', 'defendProb'];
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

<table border="1">
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
					{@const action = getAction(i, j)}
					{#if action}
						<td
							on:click={() => toggle(action, 'attacker')}
							on:contextmenu|preventDefault={() => toggle(action, 'defender')}
							class="noselect"
							class:attacking="{state.attacking.has(action.key)}"
							class:defending="{state.defending.has(action.key)}"
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
	th,
	td {
		padding: 5px;
		white-space: pre-line;
	}

	.inactive {
		background-color: gray;
	}

	.attacking {
		background-color: lightcoral;
	}

	.defending {
		background-color: #2c50c4;
	}

	.attacking.defending {
		background: linear-gradient(to right bottom, lightcoral 50%, #2c50c4 50%);
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
