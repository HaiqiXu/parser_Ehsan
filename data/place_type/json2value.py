f = open('filtered-place-types.json', 'r')
out = open('extract_result', 'w')
for line in f.readlines():
    line = line.split('"')

    if len(line) > 1 and line[1] == "value":
        out.write(line[3].lower() + '\n')
