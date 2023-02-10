# size of file in bytes
size = 2500000

out_str = ''

while len(out_str) < size:
    out_str += 'abcdefghij'

file = open('data2.txt', 'w')
file.write(out_str)
file.close()