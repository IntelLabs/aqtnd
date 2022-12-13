DATASET A

File random_tree_dataset_A.h5 contains:
	a dataset named 'H_data' of shape (100, 20,2) containing 100 random input instances
	100 datasets names 'tree_data_{idx}' containing the outputs

dataset H_data: 
- index 1 runs over the 100 random instances, each of size (20,2)
- index 2 runs over the physical index {j} of the spin chain
- index 3 labels the stored data as follows:
	- k == 0: a vector of floats of length 19, appended with a 0
	- k == 1: a vector of floats of length 20

dataset tree_data_{idx} contains the desired output for problem H_data[idx,:,:].
- contains one array, of indeterminate size, which is the adjacency matrix for the a tree
	- the tree always has 20 leaves


DATASET B

File random_tree_dataset_B.h5 contains the same data as Dataset A, but in a different format:
	a dataset named 'H_data' of shape (100, 20, 20) containing 100 random input instances
	100 datasets names 'tree_data_{idx}' containing the outputs

dataset H_data: 
- index 1 runs over the 100 random instances, each of size (20,20)
- the k == 1 vector from Dataset A is stored on the diagonal
- the k == 0 vector from Dataset A is stored directly above the diagonal, and duplicated directly below the diagonal

datasets tree_data_{idx} are identical to Dataset A
