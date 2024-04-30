# Solving Royal Game of Ur

This repository is a simple and minimal implementation of value iteration to solve Royal Game of Ur (RGU). All you need to get started is an installation of [Julia](https://julialang.org/downloads/) (a very fast and lightweight scientific programming language).

## Contents

`main.jl` contains all the code necessary to solve RGU for a given number of pieces ${N\leq7}$. If you wish to solve RGU for more than seven pieces, a state representation larger than 32 bits is required, which is not yet implemented here.