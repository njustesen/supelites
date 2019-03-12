import numpy as np
import json
import pickle


def levenshtein_slow(seq1, seq1_len, seq2, seq2_len):
    cost = 0

    # base case: empty strings
    if seq1_len == 0:
        return seq2_len
    if seq2_len == 0:
        return seq1_len

    # test if last characters of the strings match
    if seq1[seq1_len-1] == seq2[seq2_len-1]:
        cost = 0
    else:
        cost = 1

    # return minimum of delete char from s, delete char from t, and delete char from both */
    return min(levenshtein_slow(seq1, seq1_len - 1, seq2, seq2_len    ) + 1,
               levenshtein_slow(seq1, seq1_len    , seq2, seq2_len - 1) + 1,
               levenshtein_slow(seq1, seq1_len - 1, seq2, seq2_len - 1) + cost)


def levenshtein_fast(seq1, seq2):
    size_x = len(seq1) + 1
    size_y = len(seq2) + 1
    matrix = np.zeros ((size_x, size_y))
    for x in range(size_x):
        matrix [x, 0] = x
    for y in range(size_y):
        matrix [0, y] = y

    for x in range(1, size_x):
        for y in range(1, size_y):
            if seq1[x-1] == seq2[y-1]:
                matrix [x,y] = min(
                    matrix[x-1, y] + 1,
                    matrix[x-1, y-1],
                    matrix[x, y-1] + 1
                )
            else:
                matrix [x,y] = min(
                    matrix[x-1,y] + 1,
                    matrix[x-1,y-1] + 1,
                    matrix[x,y-1] + 1
                )
    #print (matrix)
    return (matrix[size_x - 1, size_y - 1])


# Load replay data
print('Parsing replay data')
with open('data/players/player_info_TvZ_2.json', newline='') as file:
    str = file.read()
    file.close()
    replays = json.loads(str)

# Load cost table
print('Parsing cost table')
with open('data/costs/all_abilities_costs_table.json', newline='') as file:
    str = file.read()
    file.close()
    costs = json.loads(str)

# Load build orders
print('Parsing build orders')
with open('data/build_orders/build_orders_TvZ_2.json', newline='') as file:
    str = file.read()
    file.close()
    build_orders = json.loads(str)

#print(build_orders)
n = len(build_orders)
n = 1000
l = 10
D = np.zeros((n, n))

int_build_orders = []
keys = list(costs.keys())
for build_order in list(build_orders.values())[:n]:
    int_build_order = []
    l_b = min(l, len(build_order))
    for build in build_order[:l_b]:
        int_build_order.append(keys.index(build))
    int_build_orders.append(int_build_order)

'''
i = 0
for player_id, build_order in build_orders.items():
    if 100 <= i <= 110:
        print(player_id)
        print(build_order[:8])
    i += 1
'''

pickle.dump(int_build_orders, open( "data/distance_matrix/TvZ_build_orders.p", "wb" ) )

i = 0
for y in range(n):
    print(i, "/", n)
    for x in range(n):
        l_x = min(l, len(int_build_orders[x]))
        l_y = min(l, len(int_build_orders[y]))
        distance = levenshtein_fast(int_build_orders[x][:l_x], int_build_orders[y][:l_y])
        #distance = levenshtein_slow(int_build_orders[x][:10], 10, int_build_orders[y][:10], 10)

        #print(distance)
        D[y][x] = distance
        D[x][y] = distance
    i+=1

print("Storing distance matrix")
pickle.dump(D, open( "data/distance_matrix/TvZ.p", "wb" ) )
