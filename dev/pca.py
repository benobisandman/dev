import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
from macro.wrangling import expanding_normalisation


# possible Diverging color map: ['PiYG', 'PRGn',  'RdBu', 'coolwarm']


def PCA_analysis(
    df_wide: pd.DataFrame,
    df_wide_pca: pd.DataFrame,
    nb_components_1: int,
    nb_component_2: int,
    level_sub_comp: str,
):
    """This function conducts a PCA analysis for a given dataframe. The input df must be in a long format,
    including the aggregated value and the different component that compose this indicator. All values must be normalized/weighted

    Args:
        df_wide (pd.DataFrame): wide format dataframe with components in columns, including aggregated component
        df_wide_PCA (pd.DataFrame): wide format dataframe with components in columns, without aggregated component
        nb_components_1 (int): number of components to include in the variance explanation graph
        nb_component_2 (int): number of component to include in the explanaion of each PC
        level_sub_comp (str): level at whitch the PCA is conducted, included in the title
    """
    if len(df_wide) < 40:
        try:
            correlation_map(df_wide)
        except Exception as error:
            print(error)
    else:
        print("Dataframe too large to display correlation map")
    pca_function(df_wide_pca, nb_components_1, level_sub_comp)
    weight_pca(df_wide_pca, nb_component_2)


def correlation_map(df: pd.DataFrame):
    """Plot correlation map

    Args:
        df (pd.DataFrame): Dataframe, wide format, including the aggregated index
    """
    corr_matrix = df.corr()

    mask = np.zeros_like(corr_matrix, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True
    f, ax = plt.subplots(figsize=(11, 15))

    # vmin = corr_matrix.min().min()
    # vmax = corr_matrix.max().max()

    heatmap = sns.heatmap(
        corr_matrix,
        mask=mask,
        square=True,
        linewidths=0.5,
        cmap="PiYG",
        cbar_kws={"shrink": 0.4, "ticks": [-1, -0.5, 0, 0.5, 1]},
        vmin=-1,
        vmax=1,
        annot=True,
        annot_kws={"size": 12},
    )

    # add the column names as labels
    ax.set_yticklabels(corr_matrix.columns, rotation=0)
    ax.set_xticklabels(corr_matrix.columns)

    sns.set_style({"xtick.bottom": True}, {"ytick.left": True})


def pca_function(df: pd.DataFrame, nb_components: int, level_sub_comp: str):
    """Conducts PCA and displays the bar graph of variance explanation by components

    Args:
        df (pd.DataFrame): datafarme in wide format, with components in columns (excluding aggregated component)
        nb_components (int): number of principal components to include in the plot
        level_sub_comp (str): level at whitch the PCA is conducted, included in the title

    Returns: Graph with the variance explained by the principal components
    """
    # compute PCA
    pca_funct = PCA(n_components=nb_components)
    pca_funct.fit(df).transform(df)

    # Graph Explained variance ratio by PCs
    plt.figure(figsize=(10, 5))
    plt.bar(
        list(range(1, nb_components + 1)), pca_funct.explained_variance_ratio_ * 100
    )
    plt.ylabel("Explained variance (in %) with", fontsize=12)
    plt.xlabel("Principal component", fontsize=12)
    plt.title(
        "Explained variance ratio PCA with weighted inflation \n" + level_sub_comp,
        fontsize=15,
    )
    for i in range(nb_components):
        plt.annotate(
            str(round(pca_funct.explained_variance_ratio_[i] * 100, 2)),
            (i + 0.75, pca_funct.explained_variance_ratio_[i] * 100 + 0.5),
            fontsize=10,
        )


def weight_pca(df: pd.DataFrame, nb_comp: int):
    """Displays a graph with the importance of each component (variable) in the main Principal Components

    Args:
        df (pd.DataFrame): datafarme in wide format, with components in columns (excluding aggregated component)
        nb_comp (int): number of principal components to include in the plot

    Returns:
    """
    pca_out = PCA().fit(df)
    pca_out.explained_variance_ratio_
    loadings = pca_out.components_
    num_pc = pca_out.n_features_
    pc_list = ["PC" + str(i) for i in list(range(1, num_pc + 1))]
    loadings_df = pd.DataFrame.from_dict(dict(zip(pc_list, loadings)))
    loadings_df["variable"] = df.columns.values
    loadings_df = loadings_df.set_index("variable")
    fig_len = len(df.columns)

    vmax = loadings_df.max().max()
    vmin = loadings_df.min().min()
    f_min = nb_comp * 2
    f_max = fig_len * 1.5
    plt.figure(figsize=(f_min, f_max))
    ax = sns.heatmap(
        loadings_df.iloc[:, :nb_comp],
        annot=True,
        cmap="PiYG",
        vmin=vmin,
        vmax=vmax,
        annot_kws={"size": 12},
    )
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=12)
    plt.yticks(rotation=0)

    plt.show()


def PCA_95(
    X: pd.DataFrame,
    scale: bool = True,
    scale_min_window: int = 5,
    lower_quantile: float = 0.05,
    upper_quantile: float = 0.95,
) -> np.array:
    """Return the principal components needed to explain 95% of the variance of X.


    Args:
        X (pd.DataFrame): Dataset to perform principal component analysis
        scale (bool, optional): Perform expanding normalization before principal compoenent analysis.
            Defaults to True.
        scale_min_window (int, optional): The min_window to start expanding. Defaults to 5.
        lower_quantile (float, optional): Lower quantile for outliers exclusion in scaling. Defaults to 0.05.
        upper_quantile (float, optional): Upper quantile for outliers exclusion in scaling. Defaults to 0.95.


    Returns:
        np.array: The cumusum of explained variance percentage
    """
    if scale:
        X = expanding_normalisation(
            X, scale_min_window, lower_quantile, upper_quantile
        )[0].dropna()

    pca_95 = PCA(n_components=0.95).fit(X)

    return np.cumsum(pca_95.explained_variance_ratio_ * 100)

