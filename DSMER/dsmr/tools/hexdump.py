import sys
p=sys.argv[1]
b=open(p,'rb').read(200)
print('BYTES:', b)
print('HEX:', b.hex())
