from pickleset import PickleDataset

# BENCHMARK CODE
from datetime import datetime
from random import seed
from random import randint

pds = PickleDataset(train_size=5, test_size=2, max_saved_chunks=8)
seed(1)
now = datetime.now()
for i in range(200):
    rand_index = randint(0, pds.__len__())
    pds.sliding_window(rand_index)
then = datetime.now()
print("Time taken = ", then - now)


"""
###BENCHMARK RESULTS:::::::::::::::###

GET ITEM

BENCHMARKING @200 iterations of __getitem__(), random indexes, 16 pickles, total 86,351,806 rows
nCHUNKS,    RAM USED (peak),    RUN TIME (s)
1           0.64GB              41.387061
2           1.10GB              38.822287
4           1.89GB              35.919033
8           3.63GB              24.747721
10          4.88GB              18.945788
12          5.91GB              14.359443
14          6.30GB              09.824801
16          8.07GB              04.012896
"""
