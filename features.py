import numpy as np


class FeatureGrid:

    def __init__(self, feature_a, feature_b, player_dict, k=10):
        self.count_grid = np.zeros((k, k))
        self.val_grid = np.zeros((k, k))
        self.feature_a = feature_a
        self.feature_b = feature_b
        a_step = (feature_a.range[1] - feature_a.range[0]) / k
        b_step = (feature_b.range[1] - feature_b.range[0]) / k
        for a in range(k):
            a_min = feature_a.range[0] + a_step * a
            a_max = a_min + a_step
            for b in range(k):
                b_min = feature_b.range[0] + b_step * b
                b_max = b_min + b_step
                val = []
                for player_id in feature_a.axis:
                    pos_a = feature_a.axis[player_id]
                    pos_b = feature_b.axis[player_id]
                    if a_min <= pos_a <= a_max and b_min <= pos_b <= b_max:
                        self.count_grid[k-a-1][b] += 1
                        val.append(player_dict[player_id].features[0])
                self.val_grid[k-a-1][b] = np.mean(val)
        self.k = k
        self.title = feature_a.title + " & " + feature_b.title


class FeatureDim:

    def __init__(self, title, features, mmrs, keep_as_max=False, use_std=False):
        self.title = title
        self.keep_as_max = keep_as_max
        self.use_std = use_std
        self.axis = []
        self.not_present = 0
        self.range = None
        for feature, mmr in zip(features, mmrs):
            self.axis.append([feature, mmr])

        x = [elm[0] for elm in self.axis if elm[0] is not np.nan]
        if len(x) > 0:
            # Compute the range
            self.range = self.set_range()
            # Keep discarded by setting to max
            if keep_as_max:
                for i in range(len(self.axis)):
                    if self.axis[i][0] is np.nan:
                        self.axis[i][0] = self.range[1]
            # Squeeze in between range
            if use_std:
                for i in range(len(self.axis)):
                    if self.axis[i][0] < self.range[0]:
                        self.axis[i][0] = self.range[0]
                    elif self.axis[i][0] > self.range[1]:
                        self.axis[i][0] = self.range[1]

    def normalize(self):
        for elm in self.axis:
            elm[0] = (elm[0] - self.range[0]) / (self.range[1] - self.range[0])

    def representation(self):
        return len(self.axis) / (len(self.axis) + self.not_present)

    def set_range(self):
        x = [elm[0] for elm in self.axis if elm[0] is not np.nan]
        if self.use_std:
            m = np.mean(x)
            std = np.std(x)
            return [m - std, m + std]
        else:
            return [np.min(x), np.max(x)]
