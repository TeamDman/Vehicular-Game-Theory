<script lang="ts">
	import type { Action, Board } from 'src/app';

	export let setBoard: (board: Board) => void;

	let content = '';

	function onClick() {
		const pythonJson = JSON.parse(content);
		const board = pythonJson.map(
			(x: any) =>
				({
					comp: x.attack.comp,
					vuln: x.attack.vuln,
					attackCost: x.attackerCost,
					defendCost: x.defenderCost,
					attackProb: x.attackerProb,
					defendProb: x.defenderProb,
					risk: x.risk,
					severity: x.severity
				} as Action)
		);
		setBoard(board);
	}
</script>

<label>
    Python JSON: 
    <textarea bind:value={content} />
</label>

<button type="button" on:click={onClick}>Import</button>
