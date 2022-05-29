<script lang="ts">
    import type { State } from 'src/app';

	import { createEventDispatcher } from 'svelte';
	export let state: State;

    const dispatch = createEventDispatcher();

</script>

Attacker capacity: <input type=number bind:value={state.attackerCapacity} min=1 max=100>
<br>
Defender capacity: <input type=number bind:value={state.attackerCapacity} min=1 max=100>
<br>
<button on:click={()=>dispatch("newboard")}>New board</button>