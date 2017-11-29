'''

Simple script to convert the negex files to CSV.
Merge all annotations to form a single classification problem.

'''

import sys, re

if len(sys.argv) < 2:
    print 'Which file?'
    exit(1)

delimiter = ','

with open(sys.argv[1]) as input, open(sys.argv[1] + '.csv', 'w') as output:
    total = 0
    output.write('sentence\ttarget\tlabel\n')
    for line in input:
        total += 1
        chunks = line.split('\t')
        if len(chunks[2].split(' ')) > 4:
            continue
        print(chunks)
        output.write(chunks[2] + '\t' + chunks[1] + '\t' + chunks[3] + '\n')
