<script lang="ts">
	import { createEventDispatcher } from 'svelte';
	export let attackerCapacity: number;
	export let defenderCapacity: number;

    const dispatch = createEventDispatcher();

</script>

Attacker capacity: <input type=number bind:value={attackerCapacity} min=1 max=100>
<br>
Defender capacity: <input type=number bind:value={defenderCapacity} min=1 max=100>
<br>
<button on:click={()=>dispatch("newboard")}>New board</button>