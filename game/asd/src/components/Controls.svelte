<script lang="ts">
	import { createEventDispatcher } from 'svelte';
	export let attackerCapacity: number;
	export let defenderCapacity: number;
	export let attackerMaxCost: number;
	export let defenderMaxCost: number;
	export let attackerMinProb: number;
	export let attackerMaxProb: number;
	export let defenderMinProb: number;
	export let defenderMaxProb: number;
    const dispatch = createEventDispatcher();

</script>


<div>
	attackerMaxCost: <input type=number bind:value={attackerMaxCost} min=1 max=100>
	<br>
	defenderMaxCost: <input type=number bind:value={defenderMaxCost} min=1 max=100>
	<br>
	attackerMinProb: <input type=number bind:value={attackerMinProb} min=0.1 max=1>
	<br>
	attackerMaxProb: <input type=number bind:value={attackerMaxProb} min=0.1 max=1>
	<br>
	defenderMinProb: <input type=number bind:value={defenderMinProb} min=0.1 max=1>
	<br>
	defenderMaxProb: <input type=number bind:value={defenderMaxProb} min=0.1 max=1>
	<br>
	<button on:click={()=>dispatch("newboard")}>New board</button>
</div>

Attacker capacity: <input type=number bind:value={attackerCapacity} min=1 max=100>
<br>
<input type="range" min=1 max=100 bind:value={attackerCapacity}>
<br>
Defender capacity: <input type=number bind:value={defenderCapacity} min=1 max=100>
<br>
<input type="range" min=1 max=100 bind:value={defenderCapacity}>
<br>