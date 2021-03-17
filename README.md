### About

This repo contains the implementation of the 'Method of Four Russians' for the 'Computation
complexity' course. 

The code is structured as follows: 

- [main.py](main.py) contains 2 implementations of matrix multiplication: default (*matmul*) and 
  the four russians algorithm (*matmul4r*) with all the necessary supplementary functions;
- [tests.py](tests.py) contains the tests;
- [comparison.ipynb](comparison.ipynb) contains the speed comparison between standard and four
  russians implementations on matrices up to size (100, 100)
  

### Usage

Code: 
```python
import numpy as np
from main import matmul, matmul4r

size = 5
# All matrices must by of type np.array!
mat_a = np.random.randint(0, 2, (size, size))
mat_b = np.random.randint(0, 2, (size, size))

res_default = matmul(mat_a=mat_a, mat_b=mat_b)  # default algo, general case
res_default_binary = matmul(mat_a=mat_a, mat_b=mat_b, binary=True)  # default algo, binary case

res_4r = matmul4r(mat_a=mat_a, mat_b=mat_b)  # four russians algo, only binary
```
Tests:

In the repository root (with installed pytest in current environment) run
```shell
pytest tests.py
```
All the tests run on every commit with the help of GitHub Actions. 


### Useful links

- [louridas.github.io/](https://louridas.github.io/rwa/assignments/four-russians/) - no implementation, only 
  description
- [neerc.ifmo.ru/wiki/](http://neerc.ifmo.ru/wiki/index.php?title=Метод_четырёх_русских_для_умножения_матриц) - 
  description in russian