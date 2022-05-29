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
	<p>attackerMaxCost: </p><input type=number bind:value={attackerMaxCost} min=1 max=100>
	<br>
	<p>defenderMaxCost: </p><input type=number bind:value={defenderMaxCost} min=1 max=100>
	<br>
	<p>attackerMinProb: </p><input type=number bind:value={attackerMinProb} step=0.01 min=0.1 max=1>
	<br>
	<p>attackerMaxProb: </p><input type=number bind:value={attackerMaxProb} step=0.01 min=0.1 max=1>
	<br>
	<p>defenderMinProb: </p><input type=number bind:value={defenderMinProb} step=0.01 min=0.1 max=1>
	<br>
	<p>defenderMaxProb: </p><input type=number bind:value={defenderMaxProb} step=0.01 min=0.1 max=1>
	<br>
	<button on:click={()=>dispatch("newboard")}>New board</button>
</div>

<p>Attacker capacity: </p><input type=number bind:value={attackerCapacity} min=1 max=100>
<br>
<input type="range" min=1 max=100 bind:value={attackerCapacity}>
<br>
<p>Defender capacity: </p><input type=number bind:value={defenderCapacity} min=1 max=100>
<br>
<input type="range" min=1 max=100 bind:value={defenderCapacity}>
<br>

<style>
	input[type=number] {
		width: 50px;
	}

	p {
		width: 130px;
		display: inline-block;
	}
</style>