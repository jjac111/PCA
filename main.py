from utils import *

pca, transformed, labels, uniques = do_PCA(manually_select=False)

model = do_training(transformed, labels, split=0.2)

do_testing(pca)