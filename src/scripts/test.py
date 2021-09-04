action_list = [[] for i in range(20)]
action_part1 = [[] for i in range(4)]
action_part2 = [[] for i in range(5)]
action_part1[0] = [0, 0, 0]
action_part1[1] = [1, 0, 0]
action_part1[2] = [1, 1, 0]
action_part1[3] = [1, 1, 1]
action_part2[0] = [0, 0, 0, 0]
action_part2[1] = [1, 0, 0, 0]
action_part2[2] = [1, 1, 0, 0]
action_part2[3] = [1, 1, 1, 0]
action_part2[4] = [1, 1, 1, 1]
for i in range(4):
    for j in range(5):
        action_list[5 * i + j] = action_part1[i] + action_part2[j]
print(action_list)
