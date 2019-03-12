import json
import numpy as np
import pickle

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

# Load build orders
print('Parsing build orders')
with open('data/build_orders/TvZ_build_orders.json', newline='') as file:
    s = file.read()
    file.close()
    build_orders = json.loads(s)
n = len(build_orders)

# Extract unit counts
print("Extracting unit counts")
unit_counts_by_player = {}
unit_counts_by_unit = {key: 0 for key in costs.keys()}
for player_id, build_order in build_orders.items():
    unit_count = {key: 0 for key in costs.keys()}
    for frame, build in build_order.items():
        unit_count[build] += 1
    unit_counts_by_player[player_id] = unit_count
    for unit in unit_count:
        if unit_count[unit] > unit_counts_by_unit[unit]:
            unit_counts_by_unit[unit] = unit_count[unit]

# Save the processed data
unit_count_obj = {
    'counts_by_player': unit_counts_by_player,
    'counts_by_unit': unit_counts_by_unit
}

pickle.dump(unit_count_obj, open("data/unit_counts/TvZ_unit_counts_{}.p".format(n), "wb"))
