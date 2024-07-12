import sys
sys.path.append("CMC_utils")

import os
import itertools
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from pandarallel import pandarallel
pandarallel.initialize(progress_bar=False)

from CMC_utils import save_load

sns.set_style("whitegrid")
from matplotlib import rc
rc('font', **{'family': 'serif', 'serif': ['Cambria']})
rc('text', usetex=True)


def load_experiment_results(experiment_info: pd.DataFrame, metric: str):
    missing_percentages = [int(perc*100) for perc in save_load.load_yaml( os.path.join( experiment_info.path, "config.yaml" ) )["missing_percentages"]]
    train_test_percentages = list(itertools.product(missing_percentages, missing_percentages))
    percentages_relative_paths = [os.path.join(str(train_test_perc[0]), str(train_test_perc[1])) for train_test_perc in train_test_percentages]

    results_paths = [os.path.join(experiment_info.path, "results", train_test_perc, "balanced", "test", "set_average_performance.csv") for train_test_perc in percentages_relative_paths]
    results = pd.concat( [save_load.load_table(path, index_col=0, header=[0, 1]).loc[["test"], [(metric, "mean"), (metric, "std")]].reset_index(drop=True).assign(train_perc=train_perc, test_perc=test_perc, db=experiment_info.db, model=experiment_info.model, imputer=experiment_info.imputer) for path, (train_perc, test_perc) in zip(results_paths, train_test_percentages)], axis=0, ignore_index=True )

    results.columns = results.columns.droplevel(0)
    results.columns = ["mean_"+metric, "std_"+metric, "train_perc", "test_perc", "db", "model", "imputer"]
    results.rename_axis(None, axis=1, inplace=True)
    results = results.set_index(["db", "model", "imputer", "train_perc", "test_perc"])

    return results


def plot_performance_by_train_fraction(data: pd.DataFrame, metric: str, focus_map: dict, hue: str, color_map: dict, filename: str, output_path: str):
    data_grouped = data.groupby('train_perc')
    rowlength = data_grouped.ngroups // 2

    percentages = sorted(data.train_perc.unique())
    percentages_str = [f"{perc}%" for perc in percentages]

    fig, axs = plt.subplots(figsize=(20, 10), nrows=2, ncols=rowlength, sharex='all', sharey='all', gridspec_kw=dict(hspace=0.5))

    targets = zip(data_grouped.groups.keys(), axs.flatten())
    for i, (key, ax) in enumerate(targets):
        l1 = sns.lineplot(data=data_grouped.get_group(key), x="test_perc", y="mean_"+metric, hue=hue, markers=True, dashes=False, ax=ax, hue_order=[x for x in color_map.keys() if x in data[hue].unique()], palette=color_map, errorbar=None, linewidth=3)
        l1.set_xticks(percentages)
        l1.set_xticklabels(percentages_str)
        ax.set(xlabel='Missing in testing (\%)', ylabel=f"{metric.upper()} (\%)")
        ax.set_title(f"Missing in training: {key}\%", fontsize=24, pad=15)

        ax.xaxis.label.set_fontsize(20)
        ax.yaxis.label.set_fontsize(20)
        ax.tick_params(axis='x', labelsize=16)
        ax.tick_params(axis='y', labelsize=16)

        handles_labels = sorted(zip(*ax.get_legend_handles_labels()), key=lambda h_l: list(focus_map.keys()).index(h_l[1]))
        handles, labels = [h for h, _ in handles_labels], [l for _, l in handles_labels]
        for handle in handles:
            handle.set_linewidth(3)
        fig.legend(handles, list(map(lambda x: focus_map[x], labels)), loc='center left', bbox_to_anchor=(0.92, 0.5), fontsize=16)
        ax.get_legend().remove()
        ax.spines['bottom'].set_color("k")
        ax.spines['top'].set_color("k")
        ax.spines['right'].set_color("k")
        ax.spines['left'].set_color("k")
        ax.xaxis.set_tick_params(labelbottom=True)
        ax.yaxis.set_tick_params(labelbottom=True)
        ax.xaxis.get_label().set_visible(True)
        ax.yaxis.get_label().set_visible(True)

    plt.savefig(os.path.join(output_path, f'{filename}.svg'), format='svg')


def plot_missing_robustness(data: pd.DataFrame, metric: str, models_map: dict, models_color_map: dict, imputers_map: dict, filename: str, output_path: str):
    missing_robustness = data.groupby(by=["model", "imputer", "train_perc", "test_perc"]).agg("mean", numeric_only=True).round(2).reset_index()
    missing_robustness.model = missing_robustness.model.map(models_map)
    missing_robustness.imputer = missing_robustness.imputer.map({**imputers_map, "-": "MIA"})

    percentages = sorted(data.train_perc.unique())

    train_missing_robust = missing_robustness.loc[missing_robustness.test_perc == percentages[0]].drop( "test_perc", axis=1 )
    no_missing_table = train_missing_robust.loc[train_missing_robust.train_perc == percentages[0]].set_index(["model", "imputer", "train_perc"]).unstack()
    no_missing_table = pd.concat([no_missing_table.rename({0: x}, axis=1, level=1) for x in percentages[1:]], axis=1)
    train_missing_robust = train_missing_robust.loc[ train_missing_robust.train_perc != percentages[0]].set_index( ["model", "imputer", "train_perc"]).unstack()
    diff_table_tr = no_missing_table.sub(train_missing_robust, axis=1).div(no_missing_table).mul(100).round(2).stack().reset_index()
    diff_table_tr["imputer"] = diff_table_tr.imputer.astype("category")
    diff_table_tr.loc[:, "train_perc"] = diff_table_tr.train_perc.map(lambda x: f"train_percentage_{x}" if len(str(x)) == 2 else f"train_percentage_0{x}")
    diff_table_tr = diff_table_tr.rename({"mean_"+metric: "missing_in_train"}, axis=1).set_index(["model", "imputer", "train_perc"]).unstack().droplevel(0, axis=1)
    del train_missing_robust, no_missing_table

    test_missing_robust = missing_robustness.loc[missing_robustness.train_perc == percentages[0]].drop( "train_perc", axis=1 )
    no_missing_table = test_missing_robust.loc[test_missing_robust.test_perc == percentages[0]].set_index(["model", "imputer", "test_perc"]).unstack()
    no_missing_table = pd.concat([no_missing_table.rename({0: x}, axis=1, level=1) for x in percentages[1:]], axis=1)
    test_missing_robust = test_missing_robust.loc[test_missing_robust.test_perc != percentages[0]].set_index(["model", "imputer", "test_perc"]).unstack()
    diff_table_te = no_missing_table.sub(test_missing_robust, axis=1).div(no_missing_table).mul(100).round(2).stack().reset_index()
    diff_table_te["imputer"] = diff_table_te.imputer.astype("category")
    diff_table_te.loc[:, "test_perc"] = diff_table_te.test_perc.map(lambda x: f"test_percentage_{x}" if len(str(x)) == 2 else f"test_percentage_0{x}")
    diff_table_te = diff_table_te.rename({"mean_"+metric: "missing_in_test"}, axis=1).set_index(["model", "imputer", "test_perc"]).unstack().droplevel(0, axis=1)
    del test_missing_robust, no_missing_table

    drop_matrix = pd.concat([diff_table_tr, diff_table_te], axis=1).reset_index()

    train_perce = [f"train_percentage_{x}" if len(str(x)) == 2 else f"train_percentage_0{x}" for x in percentages[1:]]
    test_perce = [f"test_percentage_{x}" if len(str(x)) == 2 else f"test_percentage_0{x}" for x in percentages[1:]]
    drop_matrix.loc[:, "missing_in_train"] = drop_matrix.loc[:, train_perce].mean(axis=1)
    drop_matrix.loc[:, "missing_in_test"] = drop_matrix.loc[:, test_perce].mean(axis=1)
    drop_matrix["imputer"] = drop_matrix.imputer.astype("category").cat.add_categories("No imputation")

    drop_matrix.loc[drop_matrix.model == "NAIM", "imputer"] = "No imputation"

    fig = plt.figure(figsize=(10, 10))
    ax = plt.gca()

    minix = 0
    miniy = 0
    maxix = 15
    maxiy = 20
    x, y = np.meshgrid(np.arange(minix, maxix, (maxix - minix) / 1000), np.arange(miniy, maxiy, (maxiy - miniy) / 1000))
    z = x + y

    levels = np.arange(5, maxix + maxiy + 1, 5)
    ax.contour(z, cmap=sns.color_palette("Reds_r", as_cmap=True), extent=[minix, maxix, miniy, maxiy], zorder=1, levels=levels)

    markers = {'Constant': 'o', 'KNN': '^', 'MICE': 's', 'MIA': 'X'}
    models_color_map = {models_map[model]: color for model, color in models_color_map.items()}

    sns.scatterplot(data=drop_matrix.loc[drop_matrix.model != "NAIM"], x="missing_in_train", y="missing_in_test",
                    hue="model", style="imputer", style_order=list(imputers_map.values())[1:-1] + ["MIA"],
                    hue_order=models_map.values(), s=150, ax=ax, palette=models_color_map, linewidth=0.5,
                    markers=markers)

    sns.scatterplot(data=drop_matrix.loc[drop_matrix.model == "NAIM"], x="missing_in_train", y="missing_in_test",
                    hue="model", style="imputer", style_order=["No imputation"], hue_order=["NAIM"], s=200, ax=ax,
                    palette=models_color_map, linewidth=0.5, markers={'No imputation': '*'})

    ax.set(xlabel=f'{metric.upper()} drop in training with missing values',
           ylabel=f'{metric.upper()} drop in testing with missing values')

    ax.spines['bottom'].set_color("k")
    ax.spines['top'].set_color("k")
    ax.spines['right'].set_color("k")
    ax.spines['left'].set_color("k")
    ax.xaxis.label.set_fontsize(20)
    ax.yaxis.label.set_fontsize(20)
    ax.tick_params(axis='x', labelsize=16)
    ax.tick_params(axis='y', labelsize=16)

    handles_labels = ax.get_legend_handles_labels()
    style_h_l = [handles_labels[0][1:12], handles_labels[1][1:12]]
    hue_h_l = [handles_labels[0][13:-1], handles_labels[1][13:-4]]

    style_h_l[0] = [plt.scatter([], [], marker='*', color=style_h_l[0][0].get_facecolor()[0], s=50, linewidth=0.5)] + style_h_l[0][1:]

    ax.legend(style_h_l[0], style_h_l[1], loc="center", bbox_to_anchor=(1, 0.5), fontsize=16, markerscale=2)
    ax.legend(hue_h_l[0], hue_h_l[1], loc="center", bbox_to_anchor=(1, 0.5), fontsize=16, markerscale=2)

    plt.xticks(np.arange(minix, maxix + 1, 5))
    plt.yticks(np.arange(miniy, maxiy + 1, 5))

    # Remove the current legend
    ax.legend_.remove()
    # Create new legends
    plt.gca().add_artist(ax.legend(style_h_l[0], style_h_l[1], loc='upper right', title='Model', fontsize=14, title_fontsize=16, markerscale=1.5))
    plt.gca().add_artist(ax.legend(hue_h_l[0], hue_h_l[1], loc='lower right', title='Missing Strategy', fontsize=14, title_fontsize=16, markerscale=1.5))

    plt.savefig(os.path.join(output_path, f'{filename}.svg'), format='svg')


def plot_results(metric_to_plot: str, results_folder_path: str, output_figures_path: str):
    experiments_folders = sorted([folder for folder in os.listdir(results_folder_path) if folder != ".DS_Store"])

    experiments_info = pd.DataFrame([folder.replace("_with_missing_generation", "").replace("_sklearn", "").replace("no_imputation", "-").split("_") for folder in experiments_folders], columns=["db", "model", "imputer"])
    experiments_info["path"] = [os.path.join(results_folder_path, folder) for folder in experiments_folders]

    all_results = pd.concat( experiments_info.parallel_apply(load_experiment_results, metric=metric_to_plot, axis=1).values, axis=0 )
    del experiments_folders, experiments_info

    mean_results = all_results[["mean_"+metric_to_plot]].reset_index()

    mean_results_naim = mean_results.loc[ mean_results.loc[:, "model"] == "naim" ]
    mean_results_wo_naim = mean_results.loc[ mean_results.loc[:, "model"] != "naim" ]
    mean_results_wo_no_imputation = mean_results_wo_naim.loc[ mean_results_wo_naim.loc[:, "imputer"] != "-" ]

    ML_models = ["adaboost", "dt", "histgradientboostingtree", "rf", "svm", "xgboost"]
    DL_models = ["FTTransformer", "mlp", "tabnet", "TABTransformer"]

    ## COLORS definitions

    # IMPUTERS
    imputers_map = {"naim": "NAIM", "simple": "Constant", "knn": "KNN", "iterative": "MICE", "-": "-"}
    colors = sns.color_palette("hls", 5)
    colors[0] = "r"
    imputers_colorMAP = {imp: color for imp, color in zip(imputers_map.keys(), colors)}

    # MODELS
    models_map = {"naim": "NAIM", "adaboost": "Adaboost", "dt": "Decision Tree", "FTTransformer": "FTTransformer",
                  "histgradientboostingtree": "HistGradientBoost", "mlp": "MLP", "rf": "Random Forest",
                  "svm": "SVM", "tabnet": "TabNet", "TABTransformer": "TabTransformer", "xgboost": "XGBoost"}
    colors = sns.color_palette("hls", 11)
    colors[0] = "r"
    models_colorMAP = {model: color for model, color in zip(models_map.keys(), colors)}

    ## NAIM vs IMPUTERS

    imputers_average = mean_results_wo_no_imputation.drop(["db", "model"], axis=1)
    naim_average = mean_results_naim.drop(["db", "model"], axis=1)
    naim_average.imputer = naim_average.imputer.map( {"-": "naim"} )
    naim_vs_imputers = pd.concat([naim_average, imputers_average], axis=0)
    del imputers_average, naim_average

    plot_performance_by_train_fraction(naim_vs_imputers, metric_to_plot, imputers_map, "imputer", imputers_colorMAP, "naim_vs_imputers", output_figures_path)

    ## NAIM vs MODELS w IMPUTERS

    models_average = mean_results_wo_no_imputation.drop(["db", "imputer"], axis=1)
    naim_average = mean_results_naim.drop(["db", "imputer"], axis=1)
    naim_vs_models = pd.concat([naim_average, models_average], axis=0)
    del models_average, naim_average

    plot_performance_by_train_fraction(naim_vs_models, metric_to_plot, models_map, "model", models_colorMAP, "naim_vs_all", output_figures_path)

    ## NAIM vs ML MODELS w IMPUTERS

    models_average = mean_results_wo_no_imputation.loc[mean_results_wo_no_imputation.model.isin(ML_models)].drop(["db", "imputer"], axis=1)
    naim_average = mean_results_naim.drop(["db", "imputer"], axis=1)
    naim_vs_ML_models_w_imputers = pd.concat([naim_average, models_average], axis=0)
    del models_average, naim_average

    plot_performance_by_train_fraction(naim_vs_ML_models_w_imputers, metric_to_plot, models_map, "model", models_colorMAP, "naim_vs_ML", output_figures_path)

    ## NAIM vs DL MODELS w IMPUTERS

    models_average = mean_results_wo_no_imputation.loc[mean_results_wo_no_imputation.model.isin(DL_models)].drop(["db", "imputer"], axis=1)
    naim_average = mean_results_naim.drop(["db", "imputer"], axis=1)
    naim_vs_DL_models_w_imputers = pd.concat([naim_average, models_average], axis=0)
    del models_average, naim_average

    plot_performance_by_train_fraction(naim_vs_DL_models_w_imputers, metric_to_plot, models_map, "model", models_colorMAP, "naim_vs_DL", output_figures_path)

    ## NAIM vs ML MODELS w MIA

    models_average = mean_results_wo_naim.loc[mean_results_wo_naim.imputer == "-"].drop(["db", "imputer"], axis=1)
    naim_average = mean_results_naim.drop(["db", "imputer"], axis=1)
    naim_vs_ML_models_w_MIA = pd.concat([naim_average, models_average], axis=0)
    del models_average, naim_average

    plot_performance_by_train_fraction(naim_vs_ML_models_w_MIA, metric_to_plot, models_map, "model", models_colorMAP, "naim_vs_MIA", output_figures_path)

    ## Missing robustness analysis

    plot_missing_robustness(mean_results, metric_to_plot, models_map, models_colorMAP, imputers_map, "missing_robustness", output_figures_path)


if __name__ == "__main__":
    metric_to_plot = "auc"

    results_folder_path = "./outputs"

    output_figures_path = "./plots"
    if not os.path.exists(output_figures_path):
        os.makedirs(output_figures_path)

    plot_results(metric_to_plot, results_folder_path, output_figures_path)
