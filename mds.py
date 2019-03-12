from sklearn.manifold import MDS
import os.path
import matplotlib.pyplot as plt
import json
import pickle
from sklearn.cluster import DBSCAN, SpectralClustering
import numpy as np
import matplotlib.cm as cm
import seaborn as sns
import numpy as np
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import json
from mpl_toolkits.axes_grid1 import make_axes_locatable
from features import FeatureGrid, FeatureDim


#D = np.ones((10000, 10000))
n = 7683
l = 16
cache = True


def make_grid(features, mmrs, k):
    mmr_grid = np.zeros((k, k))
    count_grid = np.zeros((k, k))
    step_size = 1 / k
    for a in range(k):
        a_min = step_size * a
        a_max = a_min + step_size
        for b in range(k):
            b_min = step_size * b
            b_max = b_min + step_size
            val = []
            for i in range(len(features)):
                feature = features[i]
                mmr = mmrs[i]
                if a_min <= feature[0] <= a_max and b_min <= feature[1] <= b_max:
                    # count_grid[k - a - 1][b] += 1
                    count_grid[k - b - 1][a] += 1
                    # count_grid[b][a] += 1
                    val.append(mmr)
            # mmr_grid[k - a - 1][b] = np.mean(val)
            mmr_grid[k - b - 1][a] = np.mean(val)
            #mmr_grid[b][a] = np.mean(val)
    return mmr_grid, count_grid


def normalize_feature(feature, features):
    if type(feature) is np.ndarray:
        norm_features = []
        for i in range(len(feature)):
            min_feature = np.min(features[:,i])
            max_feature = np.max(features[:,i])
            norm_feature = (feature[i] - min_feature) / (max_feature - min_feature)
            norm_features.append(norm_feature)
        return norm_features
    min_feature = np.min(features)
    max_feature = np.max(features)
    norm_feature = (feature - min_feature) / (max_feature - min_feature)
    return norm_feature


def readable_build_order(str_build_order):
    build_order = []
    for str_build in str_build_order:
        build_order.append(builds[ord(str_build)])
    return build_order


def closest_point(point, points):
    points = np.asarray(points)
    dist_2 = np.sum((points - point)**2, axis=1)
    return points[np.argmin(dist_2)]


def plot_grid(grid, count=False):
    # plt.style.use('ggplot')

    # brewer2mpl.get_map args: set name  set type  number of colors
    # bmap = brewer2mpl.get_map('Set2', 'qualitative', 7)

    # cmap = 'plasma'
    # cmap = 'inferno'
    # cmap = 'magma'
    # cmap = 'viridis'

    plt.clf()
    ax = sns.heatmap(grid, linewidth=0.5, vmin=np.nanmin(grid), vmax=np.nanmax(grid))
    fig = ax.get_figure()

    #ax.axes.set_xlabel(grid.feature_b, fontsize=14)
    #ax.axes.set_ylabel(grid.feature_a.title, fontsize=14)
    ax.axes.set_xticklabels(["" for i in np.arange(0, 1, 1/len(grid))])
    ax.axes.set_yticklabels(["" for i in reversed(np.arange(0, 1, 1/len(grid)))])
    fig.show()
    fig.savefig('plots/feature_maps/feature_map_{}_{}'.format(("_count" if count else ""), len(grid)) + '.pdf')


def plot_pca(data, build_orders, pca=False, eps=0.3, min_samples=10, min_clusters=1):
    fig, plot = plt.subplots()
    fig.set_size_inches(4, 4)
    plt.prism()

    #plot.scatter(data[...,0][:-5], data[...,1][:-5], c=all_colors[:-5])
    #s = [100 for n in range(5)]
    #plot.scatter(data[...,0][-5:], data[...,1][-5:], c=all_colors[-5:], marker='s', s=s)

    # Clustering
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(data)

    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_

    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    print('Estimated number of clusters: %d' % n_clusters_)
    if n_clusters_ < min_clusters:
        return

    # Black removed and is used for noise instead.
    unique_labels = set(labels)
    colors = [plt.cm.Spectral(each)
              for each in np.linspace(0, 1, len(unique_labels))]
    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Black used for noise.
            col = [0, 0, 0, 1]

        class_member_mask = (labels == k)

        xy = data[class_member_mask & core_samples_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col), markersize=3, fillstyle='full', markeredgewidth=0.0)

        # print builds in cluster
        print("Cluster {} build orders".format(k))
        for point in xy[:10]:
            for i in range(len(data)):
                if np.array_equal(data[i], point):
                    print(readable_build_order(build_orders[i]))
                    break

        # Find centroids
        if len(xy) > 0:
            center = (sum(xy[:, 0]) / len(xy), sum(xy[:, 1]) / len(xy))
            centroid = closest_point(center, xy)
            centroid_idx = -1
            assert centroid in data
            for i in range(len(data)):
                if np.array_equal(data[i], centroid):
                    centroid_idx = i
                    break
            assert centroid_idx >= 0

            print("Centroid of cluster {} with color {} and position {},{}".format(k, col, centroid[0], centroid[1]))
            print(readable_build_order(build_orders[centroid_idx]))
            plt.plot(centroid[0], centroid[1], 'o', markerfacecolor=tuple(col),
                 markeredgecolor='k', markersize=8)

            bbox_props = dict(boxstyle="circle,pad=0.1", fc="white", ec="black", lw=1)
            t = plot.text(centroid[0], centroid[1], k, ha="center", va="center",
                          rotation=0,
                          size=6,
                          bbox=bbox_props)

        xy = data[class_member_mask & ~core_samples_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col), markersize=3, fillstyle='full', markeredgewidth=0.0)

    plot.set_xticks(())
    plot.set_yticks(())
    '''
    custom_lines = [Line2D([0], [0], marker='o', color='w', label='Scatter',
                          markerfacecolor='red', markersize=4),
                    Line2D([0], [0], marker='o', color='w', label='Scatter',
                          markerfacecolor='white', markeredgecolor='black', markersize=4)]
    '''
    #for i in range(len(human_levels)):
    #    plot.annotate("Level {}".format(i), (data[len(gen_levels)+i][0], data[len(gen_levels)+i][1]))
    '''
    for i in range(len(human_levels)):
        bbox_props = dict(boxstyle="circle,pad=0.1", fc="white", ec="black", lw=2)
        t = plot.text(data[len(gen_levels)+i][0], data[len(gen_levels)+i][1], i, ha="center", va="center", rotation=0,
                    size=15,
                    bbox=bbox_props)
    '''
    plt.tight_layout(pad=-0.5, w_pad=-0.5, h_pad=-0.5)
    #plot.margins(0, 0)
    #lines = ax.plot(data)
    title = "PCA" if pca else "t-SNE"
    #plt.title(title)
    #plot.legend(custom_lines, ['PCG', 'Human'], loc=1)
    #plot.legend(['Won', 'Lost', 'Human'], loc=2)
    fig.savefig("plots/mds/{}_eps-{}_sam-{}.pdf".format("pca" if pca else "t-sne", eps, min_samples), bbox_inches='tight', pad_inches=0)
    return fig


def norm(x, min_x, max_x):
    return (x - min_x) / (max_x - min_x)


def plot_pca_mmr(data, build_orders, mmrs):
    fig, plot = plt.subplots()
    fig.set_size_inches(4, 4)
    plt.prism()
    x = data[:, 0]
    y = data[:, 1]
    min_mmr = np.min(mmrs)
    max_mmr = np.max(mmrs)
    c = cm.rainbow([norm(mmr, min_mmr, max_mmr) for mmr in mmrs])
    for i in range(len(data)):
        plt.scatter(x[i], y[i], color=c[i], s=2)

    plot.set_xticks(())
    plot.set_yticks(())

    plt.tight_layout(pad=-0.5, w_pad=-0.5, h_pad=-0.5)
    fig.savefig("plots/mds/mmr.pdf".format(),
                bbox_inches='tight', pad_inches=0)
    return fig


def plot_histogram(data, build_orders, mmrs, num_bins=30):

    plt.close('all')
    plt.style.use('ggplot')
    fig = plt.figure(figsize=(8, 5))
    plt.xlabel('')
    plt.ylabel('Count')
    #plt.title(feature.title)
    plt.xlim(-2000, 2000)
    # plt.tight_layout()
    n, bins, patches = plt.hist(data, num_bins, density=False, alpha=0.75)
    norm = mpl.colors.Normalize(vmin=2000, vmax=5000)
    #cmap = 'viridis'
    bars = []
    for i in range(len(patches)):
        patch = patches[i]
        ran = [patch.xy[0] - patch._width/2, patch.xy[0] + patch._width/2]
        # data_in_patch = [data[i] for i in range(len(data)) if ran[0] <= data[i][0] <= ran[1]]
        build_orders_in_patch = [build_orders[i] for i in range(len(data)) if ran[0] <= data[i][0] <= ran[1]]
        mmrs_in_patch = [mmrs[i] for i in range(len(data)) if ran[0] <= data[i][0] <= ran[1]]
        m = cm.ScalarMappable(norm=norm, cmap=cmap)
        mean_mmr = np.mean(mmrs_in_patch)
        rgb = m.to_rgba(mean_mmr)
        print(mean_mmr)
        patch.set_facecolor(rgb)
        print("Patch {} [{} - {}]:".format(i, ran[0], ran[1]))
        for bo in build_orders_in_patch:
            print(readable_build_order(bo))
        bars.append({
            'mean_mmr': mean_mmr,
            'build_orders': build_orders_in_patch,
            'range': ran
        })
    #cbax = fig.add_axes([0.85, 0.11, .04, .78])
    divider = make_axes_locatable(plt.gca())
    cax = divider.append_axes("right", "5%", pad="3%")
    cb1 = mpl.colorbar.ColorbarBase(cax, cmap=cmap,
                                    norm=norm,
                                    orientation='vertical')
    cb1.set_label('MMR')
    #plt.grid(True)

    plt.savefig('plots/mds_1d_{}.pdf'.format(num_bins))
    plt.clf()
    return bars


build_orders = pickle.load(open('data/build_orders/TvZ_build_orders_{}_{}.p'.format(n, l), 'rb'))
mmrs = pickle.load(open('data/mmr/TvZ_mmr_{}.p'.format(n), 'rb'))
player_ids = pickle.load(open('data/player_ids/TvZ_player_ids_{}.p'.format(n), 'rb'))

dims = [2]
for dim in dims:
    if not (cache and os.path.isfile('data/transformations/trans_{}_{}_{}.p'.format(dim, n, l))):
        D = pickle.load(open('data/distance_matrix/TvZ_{}_{}.p'.format(n, l), 'rb'))
        embedding = MDS(n_components=dim, dissimilarity='precomputed')
        transformed = embedding.fit_transform(D)
        out = {
            'embedding': embedding,
            'transformed': transformed,
            'build_orders': build_orders,
            'player_ids': player_ids,
            'mmrs': mmrs
        }
        print(transformed.shape)
        pickle.dump(out, open('data/transformations/trans_{}_{}_{}.p'.format(dim, n, l), 'wb'))

    # Load cost table
    print('Parsing cost table')
    with open('data/costs/T_costs.json', newline='') as file:
        s = file.read()
        file.close()
        costs = json.loads(s)
    builds = [build for build in costs.keys()]

    transformed = pickle.load(open(f'data/transformations/trans_{dim}_{n}_{l}.p', 'rb'))

    # Save (player_id, transformation) paris
    transform_dict = {}
    normalized_features = []
    for player_id, transformation in zip(transformed['player_ids'], transformed['transformed']):
        normalized_feature = normalize_feature(transformation, transformed['transformed'])
        normalized_features.append(normalized_feature)
        transform_dict[player_id] = normalized_feature
    with open("data/features/features_{}D.json".format(dim), "w") as file:
        json.dump(transform_dict, file)

    if dim == 2:

        plot_pca_mmr(transformed['transformed'], transformed['build_orders'], transformed['mmrs'])
        plot_pca(transformed['transformed'], transformed['build_orders'], transformed['mmrs'], eps=80, min_samples=25)

        # Feature heat map
        for k in [40, 200]:
            mmr_grid, count_grid = make_grid(normalized_features, mmrs=mmrs, k=k)
            plot_grid(mmr_grid, count=False)
            plot_grid(count_grid, count=True)

    elif dim == 1:
        for bins in [10, 25, 50, 100]:
            bars = plot_histogram(transformed['transformed'], transformed['build_orders'], transformed['mmrs'], num_bins=bins)
            pickle.dump(bars, open('data/transformations/bars_{}_{}_{}.p'.format(dim, n, l), 'wb'))
