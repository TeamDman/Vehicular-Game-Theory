<script lang="ts">
    import Board from "../components/Board.svelte"
    import {generateBoard} from "../modules/board";
    import type { Action, Player, State} from "src/app";

    const board = generateBoard();
    let state: State = {
        board,
        attacking: new Set(),
        defending: new Set(),
    };

    function toggle(event: CustomEvent<{action: Action, player: Player}>) {
        console.dir(event);
        const set = event.detail.player === "attacker" ? state.attacking : state.defending;
        if (set.has(event.detail.action.key)) set.delete(event.detail.action.key);
        else set.add(event.detail.action.key);
        state = state;
    }
</script>

<Board state={state} on:toggle={toggle}/>