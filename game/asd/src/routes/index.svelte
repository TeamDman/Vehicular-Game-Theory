<script lang="ts">
    import Board from "../components/Board.svelte"
    import Stats from "../components/Stats.svelte"
    import Controls from "../components/Controls.svelte"
    import {generateBoard} from "../lib/board";
    import type { Action, Player, State} from "src/app";

    const board = generateBoard();
    let state: State = {
        board,
        attacking: new Set(),
        defending: new Set(),
    };

    function newBoard() {
        console.log(state.board[0]);
        state.board = generateBoard();
        console.log(state.board[0]);
        state = state;
    }

    function toggle(event: CustomEvent<{action: Action, player: Player}>) {
        const set = event.detail.player === "attacker" ? state.attacking : state.defending;
        if (set.has(event.detail.action.key)) set.delete(event.detail.action.key);
        else set.add(event.detail.action.key);
        state.attacking = state.attacking;
        state.defending = state.defending;
    }
</script>

<div style="display:flex">
    <div style="margin-right:5px">
        <Board state={state} on:toggle={toggle}/>
    </div>
    <div>
        <div>
            <Stats state={state}/>
        </div>
        <div style="margin-top:10px;">
            <Controls on:newboard={newBoard}/>
        </div>
    </div>
</div>