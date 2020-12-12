import numpy as np
import os

# DATASETS x METHODS x FOLDS x METRICS
gathered = np.zeros((28, 32, 5, 6))
data_indx = 0
datasets_names = []
for i, (root, dirs, files) in enumerate(os.walk("../results/gnb")):
    for filename in files:
        filepath = root + os.sep + filename
        if filepath.endswith("gnb.npy"):
            print(filename)
            datasets_names.append(filename[:-8])
            data = np.load("%s" % filepath)
            gathered[data_indx] = data
            data_indx += 1
datasets_names = np.array(datasets_names)
np.save("dataset_names_gnb", datasets_names)
print(gathered.shape)


# Przerobic 32 na 5 x 8 -> DIV x METHODS
# DATASETS x DIV x METHODS x FOLDS x METRICS
gathered_by_div = np.zeros((datasets_names.shape[0], 5, 8, 5, 6))
for data_id in range(datasets_names.shape[0]):
    # METHODS x FOLDS x METRICS
    data_gathered = gathered[data_id]
    for metric_id in range(6):
        # METHODS x FOLDS
        metric_gathered = data_gathered[:, :, metric_id]
        for fold_id in range(5):
            # METHODS
            fold_gathered = metric_gathered[:, fold_id]
            methods_indx = [2,3,4,5,6,7]
            for div_id in range(5):
                gathered_by_div[data_id, div_id, 0, fold_id, metric_id] = fold_gathered[0]
                gathered_by_div[data_id, div_id, 1, fold_id, metric_id] = fold_gathered[1]

                gathered_by_div[data_id, div_id, 2, fold_id, metric_id] = fold_gathered[methods_indx[0]]
                gathered_by_div[data_id, div_id, 3, fold_id, metric_id] = fold_gathered[methods_indx[1]]
                gathered_by_div[data_id, div_id, 4, fold_id, metric_id] = fold_gathered[methods_indx[2]]
                gathered_by_div[data_id, div_id, 5, fold_id, metric_id] = fold_gathered[methods_indx[3]]
                gathered_by_div[data_id, div_id, 6, fold_id, metric_id] = fold_gathered[methods_indx[4]]
                gathered_by_div[data_id, div_id, 7, fold_id, metric_id] = fold_gathered[methods_indx[5]]
                methods_indx = [x+6 for x in methods_indx]
            # exit()
print(gathered_by_div.shape)
np.save("gathered_gnb", gathered_by_div)
# exit()
# """
# Gather metod z preproc
# DATASETS x METHODS x FOLDS x METRICS
dataset_names = np.load("dataset_names_gnb.npy").tolist()
gathered_preproc = np.zeros((datasets_names.shape[0], 4, 5, 6))
data_indx = 0
for i, (root, dirs, files) in enumerate(os.walk("../results/gnb")):
    for filename in files:
        filepath = root + os.sep + filename
        if filepath.endswith("preproc.npy"):
            print(filename)
            data = np.load("%s" % filepath)
            gathered_preproc[data_indx] = data
            data_indx += 1
print(gathered_preproc.shape)
np.save("gathered_gnb_preproc", gathered_preproc)
# """
