input_file_path = 'com-orkut.all.cmty.txt'
output_file_path = 'com-orkut.all.cmty_unique.txt'

unique_communities = set()

with open(input_file_path, 'r') as input_file:
    for line in input_file:
        community = tuple(sorted(map(int, line.strip().split())))
        unique_communities.add(community)

with open(output_file_path, 'w') as output_file:
    for i, community in enumerate(unique_communities, start=1):
        output_file.write(f'{i}: {community}\n')