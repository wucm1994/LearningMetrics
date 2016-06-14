import matplotlib.pyplot as plt

train = []
test = []
with open('out.txt', 'r') as f:
  train, test = zip(*[map(float, x.strip().split(',')) for x in f.readlines()])

print test[140:160]
#print train, test
x = [i * 0.1 for i in range(len(train))]
train_c = plt.plot(x, train, c='r', alpha=0.8, label='train')
test_c = plt.plot(x, test,  c='b', alpha=0.8, label='test')
n = len(train) / 10
plt.xticks(range(0, n, 5))
plt.autoscale()
plt.xlabel('epochs/100')
plt.ylabel('accuracy')
plt.ylim(ymin=0.6, ymax=0.95)
plt.xlim(xmin=0, xmax=n + 1)
#plt.legend([train_c, test_c], ['train', 'test'])
plt.legend()
plt.grid(True)
plt.show()
