# Deep-Q-learning-DQN-for-job-shop
This is the python code of deep q learning method for job shop problem with keras.

# environment

python 3.6 
keras 
numpy

# Usage 

python3 dqn.py

# The stable 4*5 job shop problem 

## problem description

```python 
self.M_processing_order = np.array(
    [[1, 3, 0, 2], [0, 2, 1, 3], [3, 1, 2, 0], [1, 3, 0, 2], [0, 1, 2, 3]])
self.M_processing_time = np.array([[18, 20, 21, 17], [18, 26, 15, 16], [
    17, 18, 27, 23], [18, 21, 25, 15], [22, 29, 28, 21]])
```


## The structure of the network

![model](model.png)

## Output of the code 

``` txt
loop : 840/10000,  score: 143.0 success: 0 / 10, e: 0.014
[3, 0, 1, 2, 4, 3, 1, 0, 2, 4, 0, 3, 2, 1, 4, 3, 0, 1, 2, 4] 20
loop : 850/10000,  score: 143.0 success: 1 / 10, e: 0.014
[1, 0, 3, 1, 2, 4, 0, 3, 2, 4, 0, 3, 1, 2, 1, 4, 0, 3, 2, 1] 20
loop : 860/10000,  score: 181.0 success: 0 / 10, e: 0.013
[3, 1, 0, 4, 3, 1, 0, 2, 4, 2, 0, 3, 2, 1, 4, 3, 1, 0, 4, 2] 20
loop : 870/10000,  score: 153.0 success: 1 / 10, e: 0.012
[4, 3, 0, 3, 1, 1, 0, 4, 2, 2, 4, 0, 3, 1, 4, 2, 3, 1, 0, 4] 20
loop : 880/10000,  score: 156.0 success: 1 / 10, e: 0.012
[1, 0, 4, 3, 2, 1, 0, 4, 2, 3, 0, 3, 1, 4, 2, 1, 0, 4, 3, 0] 20
```