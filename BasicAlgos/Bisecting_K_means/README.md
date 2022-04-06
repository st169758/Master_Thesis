# K-means and Bisecting-K-means Method
Implementation of K-means and bisecting K-means method in Python
The implementation of K-means method based on the example from the book "Machine learning in Action".
I modified the codes for bisecting K-means method since the algorithm of this part shown in this book is not really correct.

The Algorithm of Bisecting -K-means:

<1>Choose the cluster with maximum SSE from a cluster list. (Regard the whole dataset as your first cluster in the list)

<2>Find 2 sub-clusters using the basic 2-means method.

<3>Repeat <2> by NumIterations(it's up to you) times and choose the 2 sub-clusters with minimum SSE.

<4>Add these 2 sub-clusters into your culster list.

<5>Repeat the process from <1>to<4> until you get K clusters in your list. 
