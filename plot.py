import glob
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import brewer2mpl
from features import *
from random import randint
import seaborn as sns
import numpy as np
import csv
import json
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pickle
import argparse
from sklearn.decomposition import PCA
from matplotlib.lines import Line2D
from matplotlib.markers import MarkerStyle
from sklearn.cluster import DBSCAN, SpectralClustering
from sklearn import metrics
import numpy as np
from sklearn.manifold import TSNE


def build_order(player):
    bo = {}
    for key, value in player.features.items():
        bo[value] = key
    sorted_bo = []
    for build in sorted(bo.keys()):
        sorted_bo.append(bo[build])
    return bo


def plot_grid(grid, count=False):
    # plt.style.use('ggplot')

    # brewer2mpl.get_map args: set name  set type  number of colors
    # bmap = brewer2mpl.get_map('Set2', 'qualitative', 7)

    # cmap = 'plasma'
    # cmap = 'inferno'
    # cmap = 'magma'
    # cmap = 'viridis'

    if count:
        ax = sns.heatmap(grid.count_grid, linewidth=0.5, vmin=0, vmax=np.max(grid.count_grid))
    else:
        ax = sns.heatmap(grid.val_grid, linewidth=0.5, vmin=np.nanmin(grid.val_grid), vmax=np.nanmax(grid.val_grid))
    fig = ax.get_figure()
    ax.axes.set_xlabel(grid.feature_b.title, fontsize=14)
    ax.axes.set_ylabel(grid.feature_a.title, fontsize=14)
    ax.axes.set_xticklabels(np.arange(int(grid.feature_b.range[0]),
                                      int(grid.feature_b.range[1]),
                                      int((grid.feature_b.range[1] - grid.feature_b.range[0])/grid.k)))
    ax.axes.set_yticklabels(reversed(np.arange(int(grid.feature_a.range[0]),
                                               int(grid.feature_a.range[1]),
                                               int((grid.feature_a.range[1] - grid.feature_a.range[0]) / grid.k))))
    fig.show()
    fig.savefig('feature_map_' + grid.title.lower() + ("_count" if count else "") + '.pdf')


def closest_point(point, points):
    points = np.asarray(points)
    dist_2 = np.sum((points - point)**2, axis=1)
    return points[np.argmin(dist_2)]


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
            print(build_orders[centroid_idx])
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


def plot_feature_histogram(feature, num_bins=30):

    plt.close('all')
    plt.style.use('ggplot')
    fig = plt.figure(figsize=(8, 5))
    plt.xlabel('Frame')
    plt.ylabel('Count')
    plt.title(feature.title)
    #plt.xlim(feature.range[0], feature.range[1])
    # plt.tight_layout()
    x = [elm[0] for elm in feature.axis if feature.range[0] <= elm[0] <= feature.range[1]]
    n, bins, patches = plt.hist(x, num_bins, density=False, alpha=0.75)
    norm = mpl.colors.Normalize(vmin=2000, vmax=5000)
    cmap = 'viridis'
    for patch in patches:
        range = [patch.xy[0] - patch._width/2, patch.xy[0] + patch._width/2]
        mmrs = [elm[1].mmr for elm in feature.axis if range[0] <= elm[0] <= range[1]]
        m = cm.ScalarMappable(norm=norm, cmap=cmap)
        rgb = m.to_rgba(np.mean(mmrs))
        patch.set_facecolor(rgb)
    #cbax = fig.add_axes([0.85, 0.11, .04, .78])
    divider = make_axes_locatable(plt.gca())
    cax = divider.append_axes("right", "5%", pad="3%")
    cb1 = mpl.colorbar.ColorbarBase(cax, cmap=cmap,
                                    norm=norm,
                                    orientation='vertical')
    cb1.set_label('MMR')
    #plt.grid(True)

    plt.savefig('plots/feature_histograms/' + feature.title.lower() + '.pdf')
    plt.clf()


class Player:

    def __init__(self, replay_id, player_idx, result, race, apm, mmr, features):
        self.replay_id = replay_id
        self.player_idx = player_idx
        self.features = features
        self.result = result
        self.race = race
        self.apm = apm
        self.mmr = mmr

    def toJSON(self):
        return json.dumps(self, default=lambda o: o.__dict__,
            sort_keys=True, indent=4)


if __name__ == "__main__":

    # Load data data
    print('Parsing replay data')
    with open('data/players/player_info_TvZ_2.json', newline='') as file:
        str = file.read()
        file.close()
        replays = json.loads(str)

    '''
    print('Parsing resources')
    resources = {}
    for filename in glob.glob('data/resources/*.json'):
        with open(filename) as file:
            str = file.read()
            file.close()
            resources[filename.split(".")[0]] = json.loads(str)
    '''
    print('Parsing unit timings')
    with open('data/timings/first_unit_timings_TvZ_2.json') as file:
        str = file.read()
        file.close()
        timings = json.loads(str)
    '''
    cache_data = {
        'replays': replays,
        'resources': resources,
        'timings': timings
    }
    pickle.dump(cache_data, 'data/cache/feature_data')
    '''

    '''        
    parser = argparse.ArgumentParser(description='RL')
    parser.add_argument('--lr', type=float, default=7e-4,
                        help='learning rate (default: 7e-4)')
    parser.add_argument('--use-cache', action='store_true', default=False,
                        help='Whether to use gae for a2c -> ppo')
    '''
    players = []
    feature_names = set([])
    for replay_id in replays.keys():
        for player_idx in ["1", "2"]:
            player_info = replays[replay_id][player_idx]
            if replay_id in timings.keys():
                features = {key: val for key, val in timings[replay_id][player_idx].items()}
                for feature_name in features.keys():
                    if feature_name not in excluded_builds:
                        feature_names.add(feature_name)
                if player_info['player_mmr'] >= 0:
                    players.append(Player(replay_id=replay_id,
                                          player_idx=player_idx,
                                          result=player_info['result'],
                                          race=player_info['race'],
                                          apm=player_info['player_apm'],
                                          mmr=player_info['player_mmr'],
                                          features=features))

    print(len(players), " players parsed.")
    print(len(feature_names), " unique features parsed.")

    # Make a grid for each feature combination
    included = []
    excluded = []
    representations = {}
    for feature_name in feature_names:
        feature = FeatureDim(players, feature_name, race="T", keep_as_max=False, use_std=True)
        representations[feature_name] = feature.representation()
        if feature.range is not None and feature.representation() >= 0.75:
            feature = FeatureDim(players, feature_name, race="T", keep_as_max=True, use_std=True)
            included.append(feature)
            # plot_feature_histogram(feature, num_bins=25)
        else:
            excluded.append(feature)

    print("---- Included feature ----")
    for feature in included:
        print(feature.title, representations[feature.title])

    # print("---- Excluded feature ----")
    # for feature in excluded:
    #     print(feature.title)

    full_representation = 0
    for player in players:
        full_representation += 1
        for feature in included:
            if feature.title not in player.features.keys():
                full_representation -= 1
                break

    print("--- Full representation ", full_representation / len(players))

    # Make dataset with normalized values
    for feature in included:
        feature.normalize()

    # Save to json
    def obj_dict(obj):
        return obj.__dict__
    json_string = json.dumps(players, default=obj_dict)
    print(json_string[:1000])
    with open('data/out/players.json', 'w') as file:
        file.write(json_string)

    '''
    # Make grid
     grid = FeatureGrid(feature_apm, feature_mmr, player_dict, k=10)
    
    for use_count in [False, True]:
        plot_grid(grid, count=use_count)
    '''

    components = []
    players_data = [elm[1] for elm in included[0].axis]
    for feature in included:
        components.append([elm[0] for elm in feature.axis])
    components = np.array(components).T
    transformed_pca = PCA(n_components=2).fit_transform(components)
    transformed_tsne = TSNE(n_components=2).fit_transform(components)
    plot_pca(transformed_pca, players_data, pca=True, eps=0.5, min_samples=100)
    #for min_samples in range(5, 8):
    #    for eps in range(4, 8):
    #        plot_pca(transformed_tsne, players_data, pca=False, eps=eps, min_samples=min_samples*10)
    plot_pca(transformed_tsne, players_data, pca=False, eps=5, min_samples=50)
