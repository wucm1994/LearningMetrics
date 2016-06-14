import sys

def compare(a, b):
  with open(a, 'r') as f:
    xa = [x.strip().split()[0] for x in f.readlines()]
  print >>sys.stderr, xa
  with open(b, 'r') as f:
    xb = [x.strip().split()[0] for x in f.readlines()]
  #print xa, xb
  for x in xa:
    print x, xb.index(x)

if __name__ == '__main__':
  compare(sys.argv[1], sys.argv[2])
