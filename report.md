# Experimental result on Fisher's Iris Data

For the configurations defined in the table below, the optimum configuration obtained is configuration 17, since it has the lowest error rate for cross validation, and the final accuracy on the test data is 96.67%. The result can be reproduced by modifying the `tConfigKey` and `tConfigVals` variables in `iris_experiment.py` and running the script. A raw output (that does not have the final accuracy) of the experiment can be found in `out.log`.

| Configuration | Learning Rate | Nodes per layer |
|:--------------|:-------------:|:---------------:|
| 1             |      0.02     |        2        |
| 2             |      0.02     |        3        |
| 3             |      0.02     |        4        |
| 4             |      0.02     |        5        |
| 5             |      0.02     |        6        |
| 6             |      0.02     |        7        |
| 7             |      0.2      |        2        |
| 8             |      0.2      |        3        |
| 9             |      0.2      |        4        |
| 10            |      0.2      |        5        |
| 11            |      0.2      |        6        |
| 12            |      0.2      |        7        |
| 13            |       1       |        2        |
| 14            |       1       |        3        |
| 15            |       1       |        4        |
| 16            |       1       |        5        |
| 17            |       1       |        6        |
| 18            |       1       |        7        |

In this experiment, we used a maximum epoch number of 10000, and 3 iterations per round. To further verify the result or find more optimum parameters, we can vary the epoch number and number of iterations, as well as introducing more parameters.
