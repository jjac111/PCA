from utils import *

pca, transformed, labels, uniques = do_PCA(manually_select=False)

model = do_training(transformed, labels, shape, split=0.2)