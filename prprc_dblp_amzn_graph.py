with open('com-lj.all.cmty.txt', 'r') as file, open('lj_gt.txt', 'w') as output_file:
    for i, line in enumerate(file, start=1):
        numbers = line.strip().split('\t')
        total = sum(map(int, numbers))
        formatted_line = ',\t'.join(numbers)
        output_line = f'{i}: [{formatted_line}]\n'
        output_file.write(output_line)
        print(f'{i}: [{formatted_line}]')
