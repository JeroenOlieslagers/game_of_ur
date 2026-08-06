module GameOfUr

using Random

export get_bases, start_state, check_bit, check_trit, how_many_home, has_home
export move_out, move_in, flip_turn, place_piece, take_piece, move_piece
export neighbours, neighbours!, has_won, pieces_on_board, player_score, pieces_left
export turn_change, get_Ps, piece_locs, locs_to_s
export get_piece_iterator, get_pieces_dict, bfs
export get_conversions, get_neigh_tensor
export advancement, h_advancement, h_0, h_rand, h_randn, h_ninf
export initialize_value, bellman_equation, iteration!, calculate_delta
export value_iteration!, solve_game!
export fast_roll, value_to_policy, simulate_game, simulate_game_with_random
export duel, duel_with_random, tournament

include("game_logic.jl")
include("search.jl")
include("matrices.jl")
include("heuristics.jl")
include("value_iteration.jl")
include("simulations.jl")

end
