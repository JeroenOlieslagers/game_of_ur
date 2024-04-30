# Solving Royal Game of Ur

This repository is an implementation of value iteration (with some tricks) to solve Royal Game of Ur (RGU). All you need to get started is an installation of [Julia](https://julialang.org/downloads/) (a very fast and lightweight scientific programming language).

## State representation

We represent RGU board states as 32-bit unsigned integers `UInt32`. We use the self-other representation which means that it is always self's turn and we don't care about who is the light and who is the dark player. In order:
- 13 bits encode the pieces on the unsafe tiles (central column). These are actually encoded as [trits](https://en.wikipedia.org/wiki/Ternary_numeral_system). Each of 8 trits can take value 0 for empty tile, 1 for self player's piece, and 2 for other player's piece. This 8-trit number is then converted to a 13-bit number. 
- 6 bits encode whether there is a piece on the safe tiles from the self player
- 6 bits encode whether there is a piece on the safe tiles from the other player
- 3 bits encode the number of pieces still at home for the self player
- 3 bits encode the number of pieces still at home for the other player

## Value iteration

If your problem can be specified as a Markov Decision Process (MDP), then value iteration can be used to obtain optimal values for each state, and hence an optimal policy. If successful, this strongly solves the problem. Value iteration works by looping over all states, and applying Bellmans equation to update the value of each state. This process repeats until the change in value is below some threshold $\theta$.

## Computational tricks



## Contents

`main.jl` contains all the code necessary to solve RGU for a given number of pieces $N\leq7$. If you wish to solve RGU for more than seven pieces, a state representation larger than 32 bits is required, which is not yet implemented here.

&nbsp; 

`game_logic.jl` contains all functions necessary to simulate a game of RGU as well as any auxiliary functions needed that are game-dependent. Key functions include:
- `start_state` Return start state for a given number of pieces `N <= 7`
- `neighbours` Return all states that can be reached for a given dice roll
- `has_won` Return true if other has won, false otherwise.
- `get_Ps` Return dice probabilities

&nbsp; 

`search.jl` implements a search over the full state space to return all possible board states in the self-other state representation. The key function is `bfs` which implements breadth-first search to return:
- A `Dict` called `visited` with keys `(m, n)` and values the `Set` of states that have `m` pieces still to be moved to the end of the board for one players, and `n` pieces for the other player. `m` is the smaller of the two numbers.
- A `Set` called `leafs` containing all terminal states where the other player has won.

&nbsp; 

`value_iteration.jl` implements the value iteration algorithm which can solve RGU up to arbitrary precision. The key function is `value_map` which returns a `Dict` called `V` with as keys board states, and as values the associated optimal value of that state.
