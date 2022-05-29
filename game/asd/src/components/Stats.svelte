<script lang="ts">
	import type { Action, Player, State } from 'src/app';
	import { toFixed } from '$lib/utils';
	import { valueFunc } from '$lib/solver';
	export let state: State;
	$: attackCost = state.board
		.filter((x) => state.attacking.has(x.key))
		.reduce((a, b) => (a += b.attackCost), 0);
	$: defendCost = state.board
		.filter((x) => state.defending.has(x.key))
		.reduce((a, b) => (a += b.defendCost), 0);

	$: total = valueFunc(state);
	$: matches = Array.from(state.attacking)
			.filter((x) => state.defending.has(x))
			.reduce((a, v) => a + 1, 0);
	$: matchPct = toFixed(matches == 0 ? 0 : matches / state.attacking.size * 100);
</script>

Total attack cost: {attackCost}
<br />
Total defend cost: {defendCost}
<br />
Total value: {total}
<br />
Matches: {matches} ({matchPct}%)
