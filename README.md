# dpclustering
The repository for my third year project on differentially private clustering. 

## Algorithms

1. DP-Lloyd. Contained within lloyd.py this implements the iterative Lloyd algorithm was privacy preserving noise based on 4 different privacy budget allocation budgets. 

2. Grid. Contained within grid.py this divides the domain of the dataset into a uniform grid based on resolution parameter M to and averages points in each grid cell to create a private synopsis of the dataset. 

4. Tree with LSH. Main body of the algorithm contained in bucket.py and hash function in lsh.py. This algorithm creates a private synopsis through recursively splitting points based on the hash prefix. 

## File Structure 

Main algorithms are contained in the files described above. 

- *experiments/datasets* contains the data used in the final report in numpy files 

- *experiments* contains mostly Jupyter notebooks which were used for experiments and to produce plots for the final report. 

- *batch compute files* contains python files used for experiments when the batch compute system was required. 

## Usage

```bash
pip install -r requirements.txt
```

Data should be a numpy array centered and normalised to [-1,1]. Can then apply functions directly to data, for example for Tree with LSH: 

```
p = Params(epsilon=1, delta=1e-6, radius=2, dimension=3, k=4)
private_points, private_weights = create_bucket_synopsis(data, p)
```
