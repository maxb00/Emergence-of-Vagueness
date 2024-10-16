param n; # num states
param k; # num signals

set V := 1 .. n; # set of states

/* PROB[i] = probability of state i */
param PROB{V};

/* REWARD[i, j] = reward when playing action j in state i */
param REWARD{V, V};

### VARIABLES ###
/*
x[i, j] = 1 if state i is in bucket j
        = 0 otherwise
NOTE: "bucket j" means that action j is played in that bucket
NOTE: based on the assumption that there exists some optimal strategy such that
      state j is always in bucket j and there can be at most one bucket j in any 
      optimal strategy (both not formally proven, but likely true).
*/
var x{V, V} >= 0, <= 1, binary; # Hess variables

### OBJECTIVE ###
maximize EP: sum{i in V} PROB[i] * sum{j in V} x[i, j] * REWARD[i, j];

### CONSTRAINTS ###
/* there are exactly k buckets */
subject to kBucketConstr:
    sum{j in V} x[j, j] = k;

/* a state can only belong to one bucket */
subject to uniqueBucketConstr{i in V}:
    sum{j in V} x[i, j] = 1;
    
/* a state cannot belong to a non-existant bucket */
subject to nonexBucketConstr{i in V, j in V}:
    x[i, j] <= x[j, j];
    
solve;

### PRINTING ###
# printf "x=\n";
# for {i in V} {
#     printf " ";
#     for {j in V} {
#         printf "%d ", x[i, j];
#     }
#     printf "\n";
# }
# printf "\n";

printf "STATES:\n";
printf "[1 2 3]   [10 11 12]   [19 20 21]\n";
printf "[4 5 6] , [13 14 15] , [22 23 24].\n";
printf "[7 8 9]   [16 17 18]   [25 26 27]\n";
printf "\n";

printf "BUCKETS:\n";
for {j in V : x[j, j] == 1} {
    printf "Bucket %d: ", j;
    for {i in V : x[i, j] == 1} {
        printf "%d ", i;
    }
    printf "\n";
}

data;
    
### SET PARAMETERS ###

/* Case: 2 - 3 - 4 - 3 */
# param n := 9;
# param k := 4;

# param PROB :=
#     1 0.0571    2 0.1248    3 0.0571
#     4 0.1248    5 0.2724    6 0.1248
#     7 0.0571    8 0.1248    9 0.0571;

# param REWARD:
#           1     2     3     4     5     6     7     8     9 :=
#     1   1.0   0.5   0.0   0.5   0.0  -0.5   0.0  -0.5  -1.0
#     2   0.5   1.0   0.5   0.0   0.5   0.0  -0.5   0.0  -0.5
#     3   0.0   0.5   1.0  -0.5   0.0   0.5  -1.0  -0.5   0.0
#     4   0.5   0.0  -0.5   1.0   0.5   0.0   0.5   0.0  -0.5
#     5   0.0   0.5   0.0   0.5   1.0   0.5   0.0   0.5   0.0
#     6  -0.5   0.0   0.5   0.0   0.5   1.0  -0.5   0.0   0.5
#     7   0.0  -0.5  -1.0   0.5   0.0  -0.5   1.0   0.5   0.0
#     8  -0.5   0.0  -0.5   0.0   0.5   0.0   0.5   1.0   0.5
#     9  -1.0  -0.5   0.0  -0.5   0.0   0.5   0.0   0.5   1.0;

/* Case: 3 - 3 - 6 - 3 */
param n := 27;
param k := 6;

param PROB :=
    1 0.0137        2 0.0298        3 0.0137
    4 0.0298        5 0.0651        6 0.0298
    7 0.0137        8 0.0298        9 0.0137
    10 0.0298       11 0.0651       12 0.0298
    13 0.0651       14 0.1422       15 0.0651
    16 0.0298       17 0.0651       18 0.0298
    19 0.0137       20 0.0298       21 0.0137
    22 0.0298       23 0.0651       24 0.0298
    25 0.0137       26 0.0298       27 0.0137;
    
param REWARD :
            1     2     3     4     5     6     7     8     9    10    11    12    13    14    15    16    17    18    19    20    21    22    23    24    25    26    27 :=
    1     1.0   0.5   0.0   0.5   0.0  -0.5   0.0  -0.5  -1.0   0.5   0.0  -0.5   0.0  -0.5  -1.0  -0.5  -1.0  -1.5   0.0  -0.5  -1.0  -0.5  -1.0  -1.5  -1.0  -1.5  -2.0
    2     0.5   1.0   0.5   0.0   0.5   0.0  -0.5   0.0  -0.5   0.0   0.5   0.0  -0.5   0.0  -0.5  -1.0  -0.5  -1.0  -0.5   0.0  -0.5  -1.0  -0.5  -1.0  -1.5  -1.0  -1.5
    3     0.0   0.5   1.0  -0.5   0.0   0.5  -1.0  -0.5   0.0  -0.5   0.0   0.5  -1.0  -0.5   0.0  -1.5  -1.0  -0.5  -1.0  -0.5   0.0  -1.5  -1.0  -0.5  -2.0  -1.5  -1.0
    4     0.5   0.0  -0.5   1.0   0.5   0.0   0.5   0.0  -0.5   0.0  -0.5  -1.0   0.5   0.0  -0.5   0.0  -0.5  -1.0  -0.5  -1.0  -1.5   0.0  -0.5  -1.0  -0.5  -1.0  -1.5
    5     0.0   0.5   0.0   0.5   1.0   0.5   0.0   0.5   0.0  -0.5   0.0  -0.5   0.0   0.5   0.0  -0.5   0.0  -0.5  -1.0  -0.5  -1.0  -0.5   0.0  -0.5  -1.0  -0.5  -1.0
    6    -0.5   0.0   0.5   0.0   0.5   1.0  -0.5   0.0   0.5  -1.0  -0.5   0.0  -0.5   0.0   0.5  -1.0  -0.5   0.0  -1.5  -1.0  -0.5  -1.0  -0.5   0.0  -1.5  -1.0  -0.5
    7     0.0  -0.5  -1.0   0.5   0.0  -0.5   1.0   0.5   0.0  -0.5  -1.0  -1.5   0.0  -0.5  -1.0   0.5   0.0  -0.5  -1.0  -1.5  -2.0  -0.5  -1.0  -1.5   0.0  -0.5  -1.0
    8    -0.5   0.0  -0.5   0.0   0.5   0.0   0.5   1.0   0.5  -1.0  -0.5  -1.0  -0.5   0.0  -0.5   0.0   0.5   0.0  -1.5  -1.0  -1.5  -1.0  -0.5  -1.0  -0.5   0.0  -0.5
    9    -1.0  -0.5   0.0  -0.5   0.0   0.5   0.0   0.5   1.0  -1.5  -1.0  -0.5  -1.0  -0.5   0.0  -0.5   0.0   0.5  -2.0  -1.5  -1.0  -1.5  -1.0  -0.5  -1.0  -0.5   0.0
    10    0.5   0.0  -0.5   0.0  -0.5  -1.0  -0.5  -1.0  -1.5   1.0   0.5   0.0   0.5   0.0  -0.5   0.0  -0.5  -1.0   0.5   0.0  -0.5   0.0  -0.5  -1.0  -0.5  -1.0  -1.5
    11    0.0   0.5   0.0  -0.5   0.0  -0.5  -1.0  -0.5  -1.0   0.5   1.0   0.5   0.0   0.5   0.0  -0.5   0.0  -0.5   0.0   0.5   0.0  -0.5   0.0  -0.5  -1.0  -0.5  -1.0
    12   -0.5   0.0   0.5  -1.0  -0.5   0.0  -1.5  -1.0  -0.5   0.0   0.5   1.0  -0.5   0.0   0.5  -1.0  -0.5   0.0  -0.5   0.0   0.5  -1.0  -0.5   0.0  -1.5  -1.0  -0.5
    13    0.0  -0.5  -1.0   0.5   0.0  -0.5   0.0  -0.5  -1.0   0.5   0.0  -0.5   1.0   0.5   0.0   0.5   0.0  -0.5   0.0  -0.5  -1.0   0.5   0.0  -0.5   0.0  -0.5  -1.0
    14   -0.5   0.0  -0.5   0.0   0.5   0.0  -0.5   0.0  -0.5   0.0   0.5   0.0   0.5   1.0   0.5   0.0   0.5   0.0  -0.5   0.0  -0.5   0.0   0.5   0.0  -0.5   0.0  -0.5
    15   -1.0  -0.5   0.0  -0.5   0.0   0.5  -1.0  -0.5   0.0  -0.5   0.0   0.5   0.0   0.5   1.0  -0.5   0.0   0.5  -1.0  -0.5   0.0  -0.5   0.0   0.5  -1.0  -0.5   0.0
    16   -0.5  -1.0  -1.5   0.0  -0.5  -1.0   0.5   0.0  -0.5   0.0  -0.5  -1.0   0.5   0.0  -0.5   1.0   0.5   0.0  -0.5  -1.0  -1.5   0.0  -0.5  -1.0   0.5   0.0  -0.5
    17   -1.0  -0.5  -1.0  -0.5   0.0  -0.5   0.0   0.5   0.0  -0.5   0.0  -0.5   0.0   0.5   0.0   0.5   1.0   0.5  -1.0  -0.5  -1.0  -0.5   0.0  -0.5   0.0   0.5   0.0
    18   -1.5  -1.0  -0.5  -1.0  -0.5   0.0  -0.5   0.0   0.5  -1.0  -0.5   0.0  -0.5   0.0   0.5   0.0   0.5   1.0  -1.5  -1.0  -0.5  -1.0  -0.5   0.0  -0.5   0.0   0.5
    19    0.0  -0.5  -1.0  -0.5  -1.0  -1.5  -1.0  -1.5  -2.0   0.5   0.0  -0.5   0.0  -0.5  -1.0  -0.5  -1.0  -1.5   1.0   0.5   0.0   0.5   0.0  -0.5   0.0  -0.5  -1.0
    20   -0.5   0.0  -0.5  -1.0  -0.5  -1.0  -1.5  -1.0  -1.5   0.0   0.5   0.0  -0.5   0.0  -0.5  -1.0  -0.5  -1.0   0.5   1.0   0.5   0.0   0.5   0.0  -0.5   0.0  -0.5
    21   -1.0  -0.5   0.0  -1.5  -1.0  -0.5  -2.0  -1.5  -1.0  -0.5   0.0   0.5  -1.0  -0.5   0.0  -1.5  -1.0  -0.5   0.0   0.5   1.0  -0.5   0.0   0.5  -1.0  -0.5   0.0
    22   -0.5  -1.0  -1.5   0.0  -0.5  -1.0  -0.5  -1.0  -1.5   0.0  -0.5  -1.0   0.5   0.0  -0.5   0.0  -0.5  -1.0   0.5   0.0  -0.5   1.0   0.5   0.0   0.5   0.0  -0.5
    23   -1.0  -0.5  -1.0  -0.5   0.0  -0.5  -1.0  -0.5  -1.0  -0.5   0.0  -0.5   0.0   0.5   0.0  -0.5   0.0  -0.5   0.0   0.5   0.0   0.5   1.0   0.5   0.0   0.5   0.0
    24   -1.5  -1.0  -0.5  -1.0  -0.5   0.0  -1.5  -1.0  -0.5  -1.0  -0.5   0.0  -0.5   0.0   0.5  -1.0  -0.5   0.0  -0.5   0.0   0.5   0.0   0.5   1.0  -0.5   0.0   0.5
    25   -1.0  -1.5  -2.0  -0.5  -1.0  -1.5   0.0  -0.5  -1.0  -0.5  -1.0  -1.5   0.0  -0.5  -1.0   0.5   0.0  -0.5   0.0  -0.5  -1.0   0.5   0.0  -0.5   1.0   0.5   0.0
    26   -1.5  -1.0  -1.5  -1.0  -0.5  -1.0  -0.5   0.0  -0.5  -1.0  -0.5  -1.0  -0.5   0.0  -0.5   0.0   0.5   0.0  -0.5   0.0  -0.5   0.0   0.5   0.0   0.5   1.0   0.5
    27   -2.0  -1.5  -1.0  -1.5  -1.0  -0.5  -1.0  -0.5   0.0  -1.5  -1.0  -0.5  -1.0  -0.5   0.0  -0.5   0.0   0.5  -1.0  -0.5   0.0  -0.5   0.0   0.5   0.0   0.5   1.0 ;

end;
