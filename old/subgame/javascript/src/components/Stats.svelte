<script lang="ts">
	import type { Action, Board, Player } from 'src/app';
	import { toFixed } from '$lib/utils';
	import { valueFunc } from '$lib/solver';

	export let board: Board;
	export let attacking: Set<Action>;
	export let defending: Set<Action>;

	$: attackCost = board
		.filter((x) => attacking.has(x))
		.reduce((a, b) => (a += b.attackCost), 0);
	$: defendCost = board
		.filter((x) => defending.has(x))
		.reduce((a, b) => (a += b.defendCost), 0);

	$: total = valueFunc(board, attacking, defending);
	$: matches = Array.from(attacking)
			.filter((x) => defending.has(x))
			.reduce((a, v) => a + 1, 0);
	$: matchPct = toFixed(matches == 0 ? 0 : matches / attacking.size * 100);
</script>

Total attack cost: {attackCost}
<br />
Total defend cost: {defendCost}
<br />
Total value: {total}
<br />
Matches: {matches} ({matchPct}%)
