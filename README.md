#SIT_DSA_G24
Todo:

1. Algorithms:

   ~~- Advanced Graph Algorithm:~~
     ~~- change algo to A\* or yen~~
       ~~- find difference between the two and which is more efficient~~
   ~~- Data Structures:~~
     ~~- check if we use advanced data structures such as heaps for managing distances, or disjoint set unions for grouping airports~~
     ~~- if we dont, might have to implement it~~
   - Dynamic Programming:
     ~~- For multi-leg flights, consider using dynamic programming to find optimal routes considering layovers or costs~~
       - for this one we probably need to come up with how to properly calculate the cost of a path
         - **currently it is: cost is dist x 0.1 (need to find pricing)**
         - bigger plane more expensive? e.g. big plane cost > mid plane cost > small plane cost
           - i.e. 10,000km x big plane cost = $500
           - i.e. 10,000km x mid plane cost = $350
           - i.e. 10,000km x small plane cost = $200

2. Feaures:

   - Implement comprehensive error handling.
     - e.g. no routes found, etc.
   - Feedback mechanisms
     - provide clear user feedback for actions like successful operations
       - e.g. x number of routes found, etc.

3. Creativity:
   - Environmental Impact
     - feature to calculate carbon footprint of different flight routes
   - Sort flights by Distance, Cost, or Environmental Impact

# Misc

- get rid of graph intialisation page (before keying in the data)
- documentation
- presentation slides
