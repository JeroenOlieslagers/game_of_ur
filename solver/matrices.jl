
function get_conversions(visited::Dict{Tuple{Int64, Int64}, Set{UInt32}}, leaf_nodes::Set{UInt32})
    N_s = sum(length.(values(visited)))
    state_to_ind = Dict{UInt32, Int32}(union(values(visited)...) .=> zeros(Int32, N_s))
    ind_to_state = zeros(UInt32, N_s+length(leaf_nodes))
    boundaries = Dict{Tuple{Int64, Int64}, Tuple{Int32, Int32}}()
    counter = 0
    mns = get_piece_iterator(7)
    for pieces in mns
        if pieces in keys(visited)
            first_edge = counter+1
            for s in visited[pieces]
                counter += 1
                ind_to_state[counter] = s
                state_to_ind[s] = counter
            end
            boundaries[pieces] = (first_edge, counter)
        end
    end
    for s in leaf_nodes
        counter += 1
        ind_to_state[counter] = s
        state_to_ind[s] = counter
    end
    return ind_to_state, state_to_ind, boundaries
end

function get_neigh_tensor(states::Set{UInt32}, state_to_ind::Dict{UInt32, Int32})
    N_s = length(states)
    neigh_tensor = zeros(Int32, 7, 4, N_s);
    mirror_states = zeros(Int32, N_s);
    ns = zeros(UInt32, 7)
    for s in ProgressBar(states)
        #s = ind_to_state[n]
        n = state_to_ind[s]
        mirror_states[n] = state_to_ind[flip_turn(s, bs, bbs) - bs[32]]
        for d in 1:4
            neighbours!(ns, s, d, bs, bbs)
            for (i, neigh) in enumerate(ns)
                if neigh == 0
                    break
                end
                if check_bit(neigh, 32)
                    neigh_tensor[i, d, n] = -state_to_ind[neigh - bs[32]]
                else
                    neigh_tensor[i, d, n] = state_to_ind[neigh]
                end
            end
        end
    end
    return neigh_tensor, mirror_states
end