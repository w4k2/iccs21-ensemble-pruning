import numpy as np
from scipy.stats import rankdata, ranksums
from tabulate import tabulate
import pandas as pd
import matplotlib.pyplot as plt
from math import pi
import numpy as np
from matplotlib import rcParams
import seaborn as sb
rcParams["font.family"] = "monospace"
tablefmt = "latex_booktabs"
# tablefmt = "github"


# DATASETS x DIV x METHODS x FOLDS x METRICS
results = np.load("gathered_gnb.npy")
# DATASETS x DIV x METHODS x METRICS
results = np.mean(results, axis=3)
# DATASETS x DIV x METHODS -> Only BAC
results = results[:, :, : ,0]
# print(results.shape)

diversity_measures = ["E", "k", "KW", "Dis", "Q"]
clfs = ["GNB-CL2", "GNB-CL3", "GNB-CL4", "GNB-CL5", "GNB-CL6", "GNB-CL7"]
metrics = ["BAC", "G-mean", "F1", "Precision", "Recall", "Specificity"]

dataset_names = np.load("dataset_names_gnb.npy")
# print(dataset_names)
# for data_id, dataset in enumerate(dataset_names):
#     print("%s" % dataset)
#     # DIV x METHODS
#     data_results = results[data_id]
#     for div_id in range(5):
#         # METHODS without state-of-the-art
#         div_results = data_results[div_id][2:]
#         print(div_results, div_results.shape)
    # exit()
"""
Parametryzacja liczby klastrów dla kazdej miary div pod względem BAC
"""
print("\nEXPERIMENT 1 - PARAMETRIZATION\n")
t = []
best_clusters = []
for div_id, div in enumerate(diversity_measures):
    # print("\n######## %s diversity measure ########\n" % div)
    # DATASETS x METHODS without state-of-the-art
    div_results = results[:, div_id, :]
    div_global_table = []
    for data_id, dataset in enumerate(dataset_names):
        # print("%s" % dataset)
        # METHODS
        data_results = div_results[data_id][2:]
        div_global_table.append(data_results)
        # print(data_results)
    div_global_table = np.array(div_global_table)
    # print(div_global_table, div_global_table.shape)
    mean_scores = np.mean(div_global_table, axis=0)
    # print(mean_scores)

    # Ranking
    ranks = []
    for ms in div_global_table:
        ranks.append(rankdata(ms).tolist())
    ranks = np.array(ranks)
    mean_ranks = np.mean(ranks, axis=0)
    best_clusters.append(np.argmax(mean_ranks)+2)
    # print("\nRanks:\n", ranks)
    # print("\nMean ranks:\n", )

    alpha = .05
    length = len(clfs)

    s = np.zeros((length, length))
    p = np.zeros((length, length))

    for i in range(length):
        for j in range(length):
            s[i, j], p[i, j] = ranksums(ranks.T[i], ranks.T[j])
    _ = np.where((p < alpha) * (s > 0))
    conclusions = [list(1 + _[1][_[0] == i])
                   for i in range(length)]

    t.append(["%s" % div] + ["%.3f" % v for v in mean_ranks])

    # t.append([''] + [", ".join(["%i" % i for i in c])
    #                  if len(c) > 0 else nc
    #                  for c in conclusions])
    t.append([''] + [", ".join(["%i" % i for i in c])
                     if len(c) > 0 and len(c) < len(clfs)-1 else ("all" if len(c) == len(clfs)-1 else "---")
                     for c in conclusions])

    # print(t)
print(tabulate(t, headers=clfs, tablefmt=tablefmt))
print("\nSELECTED NUMBER OF CLUSTERS")
print("E: %i, k: %i, KW: %i, DIS: %i, Q: %i" % (best_clusters[0], best_clusters[1], best_clusters[2], best_clusters[3], best_clusters[4]))
# exit()

"""
Porownanie ze state of the art ,ranking, wszystkie metryki
"""
# DATASETS x DIV x METHODS x FOLDS x METRICS
results = np.load("gathered_gnb.npy")
# DATASETS x DIV x METHODS x METRICS
results = np.mean(results, axis=3)

clfs = ["GNB-MV", "GNB-SACC", "GNB-E6", "GNB-k6", "GNB-KW6", "GNB-DIS4", "GNB-Q5"]
clfs = ["MV", "SACC", "RSMO-E6", "RSMO-k6", "RSMO-KW6", "RSMO-DIS4", "RSMO-Q5"]



print("\nEXPERIMENT 2 - S-O-T-A COMPARISON\n")
t = []
plot_mean_ranks = []
t_mean_scores = []
for metric_id ,metric in enumerate(metrics):
    # print("######## %s ########\n" % metric)
    # DATASETS x DIV x METHODS
    metric_results = results[:, :, :, metric_id]
    metric_global_table = []
    for data_id, dataset in enumerate(dataset_names):
        # print("%s" % dataset)
        # DIV x METHODS
        data_results = metric_results[data_id]
        # print(data_results, data_results.shape)
        selected_methods = np.zeros((len(clfs)))
        for div_id, div in enumerate(diversity_measures):
            # print("%s" % div)
            # METHODS
            div_results = data_results[div_id]
            selected_methods[0] = div_results[0]
            selected_methods[1] = div_results[1]
            selected_methods[div_id+2] = div_results[best_clusters[div_id]]
        # print(selected_methods)
        metric_global_table.append(selected_methods)
    metric_global_table = np.array(metric_global_table)
    mean_scores = np.mean(metric_global_table, axis=0)
    # print(mean_scores)
    t_mean_scores.append(mean_scores)
    t_mean_scores.append(np.std(metric_global_table, axis=0))

    # Ranking
    ranks = []
    for ms in metric_global_table:
        ranks.append(rankdata(ms).tolist())
    ranks = np.array(ranks)
    mean_ranks = np.mean(ranks, axis=0)
    plot_mean_ranks.append(mean_ranks)
    # print("\nRanks:\n", ranks)
    # print("\nMean ranks:\n", mean_ranks)

    alpha = .05
    length = len(clfs)

    s = np.zeros((length, length))
    p = np.zeros((length, length))

    for i in range(length):
        for j in range(length):
            s[i, j], p[i, j] = ranksums(ranks.T[i], ranks.T[j])
    _ = np.where((p < alpha) * (s > 0))
    conclusions = [list(1 + _[1][_[0] == i])
                   for i in range(length)]

    t.append(["%s" % metric] + ["%.3f" % v for v in mean_ranks])

    # t.append([''] + [", ".join(["%i" % i for i in c])
    #                  if len(c) > 0 else nc
    #                  for c in conclusions])
    t.append([''] + [", ".join(["%i" % i for i in c])
                     if len(c) > 0 and len(c) < len(clfs)-1 else ("all" if len(c) == len(clfs)-1 else "---")
                     for c in conclusions])

    # print(t)
print(tabulate(t, headers=clfs, tablefmt=tablefmt))
std_metrics = np.array(["BAC", "", "G-mean", "", "F1", "", "Precision", "", "Recall", "", "Specificity", ""]).reshape(-1,1)
t_mean_scores = np.concatenate((std_metrics, t_mean_scores), axis=1)
print(tabulate(t_mean_scores, headers=clfs, tablefmt=tablefmt, floatfmt=".3f"))
# Plot radar diagram

pal = sb.color_palette("rocket")
# print(pal.as_hex())
colors = ["black", "black", '#701f57', '#ad1759', '#e13342', '#f37651', '#f6b48f']

# colors = ["black", "black", '#f6b48f', '#701f57', '#ad1759', '#e13342', '#f37651']

# colors = [(0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0.9, 0), (0, 0.9, 0), (0, 0, 0.9), (0, 0, 0.9)]
# styl linii
ls = ["-", "--", "--", "--", "--", "--", "--"]
# grubosc linii
lw = [1, 1, 1, 1, 1, 1, 2]

plot_mean_ranks = np.array(plot_mean_ranks).T
radar_dt = pd.DataFrame(data=plot_mean_ranks, columns=metrics)
groups = pd.DataFrame({'group': clfs})
df = groups.join(radar_dt)
# print(df)

categories = list(df)[1:]
N = len(categories)

angles = [n / float(N) * 2 * pi for n in range(N)]
angles += angles[:1]

# Initialise the spider plot
ax = plt.subplot(111, polar=True)

# If you want the first axis to be on top:
ax.set_theta_offset(pi / 2)
ax.set_theta_direction(-1)

# No shitty border
ax.spines["polar"].set_visible(False)

# Draw one axe per variable + add labels labels yet
plt.xticks(angles, categories)

# Adding plots
for i in range(7):
    values = df.loc[i].drop("group").values.flatten().tolist()
    values += values[:1]
    # print(values)
    values = [float(i) for i in values]
    ax.plot(
        angles, values, label=df.iloc[i, 0], c=colors[i], ls=ls[i], lw=lw[i],
    )
# Add legend
plt.legend(
    loc="lower center",
    ncol=4,
    columnspacing=1,
    frameon=False,
    bbox_to_anchor=(0.5, -0.3),
    fontsize=6,
)

# Add a grid
plt.grid(ls=":", c=(0.7, 0.7, 0.7))

# Add a title
plt.title("GNB" , size=8, y=1.08)
plt.tight_layout()

# Draw labels
a = np.linspace(0, 1, 8)
plt.yticks(
        [0,1, 2, 3, 4, 5, 6, 7],
        ["0", "1", "2", "3", "4", "5", "6", "7"],
        fontsize=6,
    )
plt.ylim(0.0, 7.0)
plt.gcf().set_size_inches(4, 3.5)
plt.gcf().canvas.draw()
angles = np.rad2deg(angles)

ax.set_rlabel_position((angles[0] + angles[1]) / 2)

har = [(a >= 90) * (a <= 270) for a in angles]

for z, (label, angle) in enumerate(zip(ax.get_xticklabels(), angles)):
    x, y = label.get_position()
    # print(label, angle)
    lab = ax.text(
        x, y, label.get_text(), transform=label.get_transform(), fontsize=6,
    )
    lab.set_rotation(angle)

    if har[z]:
        lab.set_rotation(180 - angle)
    else:
        lab.set_rotation(-angle)
    lab.set_verticalalignment("center")
    lab.set_horizontalalignment("center")
    lab.set_rotation_mode("anchor")

for z, (label, angle) in enumerate(zip(ax.get_yticklabels(), a)):
    x, y = label.get_position()
    # print(label, angle)
    lab = ax.text(
        x,
        y,
        label.get_text(),
        transform=label.get_transform(),
        fontsize=4,
        c=(0.7, 0.7, 0.7),
    )
    lab.set_rotation(-(angles[0] + angles[1]) / 2)

    lab.set_verticalalignment("bottom")
    lab.set_horizontalalignment("center")
    lab.set_rotation_mode("anchor")

ax.set_xticklabels([])
ax.set_yticklabels([])

plt.savefig("gnb_radar12.png", bbox_inches='tight', dpi=300)
plt.savefig("gnb_radar12.eps", bbox_inches='tight', dpi=300)
plt.close()
# exit()
# exit()
print("\nEXPERIMENT 3 - PREPROC COMPARISON\n")
# DATASETS x METHODS x FOLDS x METRICS
results_pre = np.load("gathered_gnb_preproc.npy")
# DATASETS x METHODS x METRICS
results_pre = np.mean(results_pre, axis=2)
# print(results_pre.shape)
# exit()
# DATASETS x DIV x METHODS x FOLDS x METRICS
results = np.load("gathered_gnb.npy")
# print()
# DATASETS x DIV x METHODS x METRICS
results = np.mean(results, axis=3)
# Get only Q5
# DATASETS x METRICS
results_q5 = results[:, 4, 5, :]
clfs = ["ROS", "SMOTE", "SVM", "B2", "RSMO-Q5"]
t_mean_scores = []
plot_mean_ranks = []
t = []
for metric_id ,metric in enumerate(metrics):
    # print("######## %s ########\n" % metric)
    # DATASETS x DIV x METHODS
    metric_results = results_pre[:, :, metric_id]
    # DATASET
    get_q50 = results_q5[:, metric_id]
    metric_global_table = []
    for data_id, dataset in enumerate(dataset_names):
        # print("%s" % dataset)
        # DIV x METHODS
        data_results = metric_results[data_id]
        # print(get_q50[data_id])
        data_all = np.concatenate((data_results, [get_q50[data_id]]))
        # print(data_results, data_results.shape)
        metric_global_table.append(data_all)
    metric_global_table = np.array(metric_global_table)
    # print(metric_global_table)
    # exit()
    mean_scores = np.mean(metric_global_table, axis=0)
    # print(mean_scores)
    t_mean_scores.append(mean_scores)
    t_mean_scores.append(np.std(metric_global_table, axis=0))

    # Ranking
    ranks = []
    for ms in metric_global_table:
        ranks.append(rankdata(ms).tolist())
    ranks = np.array(ranks)
    mean_ranks = np.mean(ranks, axis=0)
    plot_mean_ranks.append(mean_ranks)
    # print("\nRanks:\n", ranks)
    # print("\nMean ranks:\n", mean_ranks)

    alpha = .05
    length = len(clfs)

    s = np.zeros((length, length))
    p = np.zeros((length, length))

    for i in range(length):
        for j in range(length):
            s[i, j], p[i, j] = ranksums(ranks.T[i], ranks.T[j])
    _ = np.where((p < alpha) * (s > 0))
    conclusions = [list(1 + _[1][_[0] == i])
                   for i in range(length)]

    t.append(["%s" % metric] + ["%.3f" % v for v in mean_ranks])

    # t.append([''] + [", ".join(["%i" % i for i in c])
    #                  if len(c) > 0 else nc
    #                  for c in conclusions])
    t.append([''] + [", ".join(["%i" % i for i in c])
                     if len(c) > 0 and len(c) < len(clfs)-1 else ("all" if len(c) == len(clfs)-1 else "---")
                     for c in conclusions])

    # print(t)
print(tabulate(t, headers=clfs, tablefmt=tablefmt))
std_metrics = np.array(["BAC", "", "G-mean", "", "F1", "", "Precision", "", "Recall", "", "Specificity", ""]).reshape(-1,1)
t_mean_scores = np.concatenate((std_metrics, t_mean_scores), axis=1)
print(tabulate(t_mean_scores, headers=clfs, tablefmt=tablefmt, floatfmt=".3f"))

pal = sb.color_palette("rocket")
# print(pal.as_hex())
colors = ['#701f57', '#ad1759', '#e13342', '#f37651', '#f6b48f']

colors = ["black", "black", 'black', 'black', "red"]

# colors = [(0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0.9, 0), (0, 0.9, 0), (0, 0, 0.9), (0, 0, 0.9)]
# styl linii
ls = ["-", "--", ":", "-.", "-"]
# grubosc linii
lw = [1, 1, 1, 1, 1.5]

plot_mean_ranks = np.array(plot_mean_ranks).T
radar_dt = pd.DataFrame(data=plot_mean_ranks, columns=metrics)
groups = pd.DataFrame({'group': clfs})
df = groups.join(radar_dt)
# print(df)

categories = list(df)[1:]
N = len(categories)

angles = [n / float(N) * 2 * pi for n in range(N)]
angles += angles[:1]

# Initialise the spider plot
ax = plt.subplot(111, polar=True)

# If you want the first axis to be on top:
ax.set_theta_offset(pi / 2)
ax.set_theta_direction(-1)

# No shitty border
ax.spines["polar"].set_visible(False)

# Draw one axe per variable + add labels labels yet
plt.xticks(angles, categories)

# Adding plots
for i in range(5):
    values = df.loc[i].drop("group").values.flatten().tolist()
    values += values[:1]
    # print(values)
    values = [float(i) for i in values]
    ax.plot(
        angles, values, label=df.iloc[i, 0], c=colors[i], ls=ls[i], lw=lw[i],
    )
# Add legend
plt.legend(
    loc="lower center",
    ncol=3,
    columnspacing=1,
    frameon=False,
    bbox_to_anchor=(0.5, -0.3),
    fontsize=6,
)

# Add a grid
plt.grid(ls=":", c=(0.7, 0.7, 0.7))

# Add a title
plt.title("GNB" , size=8, y=1.08)
plt.tight_layout()

# Draw labels
a = np.linspace(0, 1, 8)
plt.yticks(
        [0,1, 2, 3, 4, 5, 6, 7],
        ["0", "1", "2", "3", "4", "5", "6", "7"],
        fontsize=6,
    )
plt.ylim(0.0, 7.0)
plt.gcf().set_size_inches(4, 3.5)
plt.gcf().canvas.draw()
angles = np.rad2deg(angles)

ax.set_rlabel_position((angles[0] + angles[1]) / 2)

har = [(a >= 90) * (a <= 270) for a in angles]

for z, (label, angle) in enumerate(zip(ax.get_xticklabels(), angles)):
    x, y = label.get_position()
    # print(label, angle)
    lab = ax.text(
        x, y, label.get_text(), transform=label.get_transform(), fontsize=6,
    )
    lab.set_rotation(angle)

    if har[z]:
        lab.set_rotation(180 - angle)
    else:
        lab.set_rotation(-angle)
    lab.set_verticalalignment("center")
    lab.set_horizontalalignment("center")
    lab.set_rotation_mode("anchor")

for z, (label, angle) in enumerate(zip(ax.get_yticklabels(), a)):
    x, y = label.get_position()
    # print(label, angle)
    lab = ax.text(
        x,
        y,
        label.get_text(),
        transform=label.get_transform(),
        fontsize=4,
        c=(0.7, 0.7, 0.7),
    )
    lab.set_rotation(-(angles[0] + angles[1]) / 2)

    lab.set_verticalalignment("bottom")
    lab.set_horizontalalignment("center")
    lab.set_rotation_mode("anchor")

ax.set_xticklabels([])
ax.set_yticklabels([])

plt.savefig("gnb_radar22.png", bbox_inches='tight', dpi=300)
plt.savefig("gnb_radar22.eps", bbox_inches='tight', dpi=300)
plt.close()
