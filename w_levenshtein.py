import numpy as np
import json
import pickle
from weighted_levenshtein import lev, osa, dam_lev
import numpy as np
import json

ascii_index = 0

# Load replay data
print('Parsing player data')
with open('data/players/TvZ_player_info.json', newline='') as file:
    s = file.read()
    file.close()
    players = json.loads(s)

# Load cost table
print('Parsing cost table')
with open('data/costs/T_costs.json', newline='') as file:
    s = file.read()
    file.close()
    costs = json.loads(s)

keys = list(costs.keys())
insert_costs = np.ones(len(keys)+ascii_index, dtype=np.float64)
delete_costs = np.ones(len(keys)+ascii_index, dtype=np.float64)
substitute_costs = np.ones((len(keys)+ascii_index, len(keys)+65), dtype=np.float64)
transpose_costs = np.ones((len(keys)+ascii_index, len(keys)+65), dtype=np.float64)
for i in range(len(costs)):
    build = keys[i]
    cost_i = costs[build]['minerals']+costs[build]['vespene']
    insert_costs[i+ascii_index] = cost_i
    delete_costs[i+ascii_index] = cost_i
    for j in range(len(costs)):
        cost_j = costs[keys[j]]['minerals'] + costs[keys[j]]['vespene']
        substitute_costs[i+ascii_index][j+ascii_index] = cost_i + cost_j
    for j in range(len(costs)):
        cost_j = costs[keys[j]]['minerals'] + costs[keys[j]]['vespene']
        transpose_costs[i+ascii_index][j+ascii_index] = min(cost_i, cost_j)

# Load build orders
print('Parsing build orders')
with open('data/build_orders/TvZ_build_orders.json', newline='') as file:
    s = file.read()
    file.close()
    build_orders = json.loads(s)

#print(build_orders)
n = len(build_orders)
#n = 3000
l = 16

'''
int_build_orders = []
for build_order in list(build_orders.values())[:n]:
    int_build_order = []
    l_b = min(l, len(build_order))
    for build in build_order[:l_b]:
        int_build_order.append(keys.index(build))
    int_build_orders.append(int_build_order)
'''

str_build_orders = []
mmrs = []
player_ids = []
for player_id, build_order in list(build_orders.items())[:n]:
    mmr = players[player_id]['player_mmr']
    if len(build_order) < l or mmr < 0:
        continue
    mmrs.append(mmr)
    player_ids.append(player_id)
    str_build_order = ""
    l_b = min(l, len(build_order))
    sorted_build_order = {int(k): v for k, v in build_order.items()}
    for frame in sorted(list(sorted_build_order.keys()))[:l_b]:
        build = sorted_build_order[frame]
        if costs[build]['minerals'] + costs[build]['vespene'] > 0:
            c = chr(keys.index(build) + ascii_index)
            str_build_order += c
    if str_build_order != "":
        str_build_orders.append(str_build_order)
        print(str_build_order)

print(sorted(str_build_orders))
n = len(str_build_orders)

D = np.zeros((n, n))

'''
i = 0
for player_id, build_order in build_orders.items():
    if 100 <= i <= 110:
        print(player_id)
        print(build_order[:8])
    i += 1
'''

pickle.dump(str_build_orders, open("data/build_orders/TvZ_build_orders_{}_{}.p".format(n, l), "wb"))
pickle.dump(mmrs, open("data/mmr/TvZ_mmr_{}.p".format(n), "wb"))
pickle.dump(player_ids, open("data/player_ids/TvZ_player_ids_{}.p".format(n), "wb"))

i = 0
for y in range(n):
    print(i, "/", n)
    for x in range(n):
        str_build_orders[x] = str_build_orders[x]
        str_build_orders[y] = str_build_orders[y]
        if y == x:
            distance = 0
        else:
            distance = dam_lev(str1=str_build_orders[x],
                               str2=str_build_orders[y],
                               insert_costs=insert_costs,
                               delete_costs=delete_costs,
                               substitute_costs=substitute_costs,
                               transpose_costs=transpose_costs
                               )
        distance = int(distance)
        D[y][x] = distance
        D[x][y] = distance
        #print(str_build_orders[x])
        #print(str_build_orders[y])
        #print(distance)
    i += 1

print("Storing distance matrix")
pickle.dump(D, open("data/distance_matrix/TvZ_{}_{}.p".format(n, l), "wb"))