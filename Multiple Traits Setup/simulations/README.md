For these simulations, our goal initially was to simulate similar results to what we have been doing with previous setups. However, gradually we switched our attention from looking at how well the agents do (in terms of payoff) to how information content are conveyed.

Here's a breakdown of all the different versions:
- v1: simulations contain a simple plot showing how information content changes.
- v2: add information content by individual signals
- v3: add weighted and unweighted information content
- v4: add information content by each state of each signal
- v5: add a plot showing payoff over time and information content by the distance from the center state (the last two lines; "corner" show states furthest from the center, "middle" show states one step away from the center, and "center" is the center state)
- v6: add information by row/column (the last two lines) and a .txt file for a 3D case (since we didn't come up with a good way of displaying the 3D case as a .gif file)
- v7: more .txt files, the reason is because we wanted to run multiple simulations with the same parameters at the same time and take the average of all those runs
- v8: more .txt files, this time adding information by individual states for both unweighted and weighted information content
- 
