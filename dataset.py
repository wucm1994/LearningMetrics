import os
import numpy as np
import glob
import random
data_dir = '/Users/yujia/Desktop/data'
log_dir  = '/Users/yujia/Desktop/mysite/logs'

debug = False

def load_model_data(token=''):
  data = {}
  imgs = set()
  for root, dirs, files in os.walk(data_dir):
    if token not in root:
      continue
    for file in files:
      if file.endswith('.TXT'):
        name = file[:-4]
        dup = False
        if name in data:
          dup = True
        with open(os.path.join(root, file), 'r') as f:
          lines = f.readlines()
          if dup and debug:
            print map(lambda x : float(x.strip()), lines), data[name]
          try:
            data[name] = map(lambda x : float(x.strip()), lines)
          except:
            if debug:
              print root, file
      elif file.endswith('.jpg'):
        assert file not in imgs
        imgs.add(file[:-6])
  imgs = list(imgs)
  return np.array([data[x] for x in imgs if x in data], dtype='float32'), [x for x in imgs if x in data]

#print load_model_data('Rabbit')[0].shape
#exit()


def load_dataset(token='', include=True, left_bound=None, right_bound=None):
  data = {}
  imgs = set()
  for root, dirs, files in os.walk(data_dir):
    if (include and token not in root) or (not include and token in root):
      continue
    for file in files:
      if file.endswith('.TXT'):
        name = file[:-4]
        dup = False
        if name in data:
          dup = True
        with open(os.path.join(root, file), 'r') as f:
          lines = f.readlines()
          if dup and debug:
            print map(lambda x : float(x.strip()), lines), data[name]
          try:
            data[name] = map(lambda x : float(x.strip()), lines)
          except:
            if debug:
              print root, file
      elif file.endswith('.jpg'):
        assert file not in imgs
        imgs.add(file[:-6])
  #print len(data), len(imgs)
  tot = {}
  win = {}
  for logfile in glob.glob(os.path.join(log_dir, '*.txt')):
    #if not os.path.split(logfile)[-1].startswith('yujia'):
      #continue
    with open(logfile, 'r') as f:
      for line in f.readlines():
        x, y = line.strip().split()
        if x not in imgs or y not in imgs:
          #print x, y
          continue
        if x < y:
          tot[(x, y)] = tot.get((x, y), 0) + 1
          win[(x, y)] = win.get((x, y), 0) + 1
        else:
          tot[(y, x)] = tot.get((y, x), 0) + 1
  res = []
  names = tot.keys()
  random.seed(0)
  random.shuffle(names)
  good_names = []
  for x, y in names:
    assert len(data[x]) == 6, data[x]
    assert len(data[y]) == 6, data[y]
    wins = win.get((x, y), 0)
    tots = tot.get((x, y), 0)
    if abs(tots - 2 * wins) * 3 <= tots:
      pass
    elif wins * 2 >= tots:
      res.append((data[x], data[y], 1.))
      good_names.append((x, y))
    else:
      res.append((data[x], data[y], 0.))
      good_names.append((x, y))
  names = good_names
  names = names[left_bound:right_bound]
  res = res[left_bound:right_bound]
  res = zip(*res) if res else [[],[],[]]
  x1, x2, y = map(lambda x : np.array(x, dtype='float32'), res)
  return x1, x2, y, names

#train_data = load_dataset(right_bound=1080)
#test_data = load_dataset(left_bound=1080)

token = 'Bunny'
test_data = load_dataset(token, include=True)
train_data = load_dataset(token, include=False)

print 'test:', test_data[0].shape
print 'train:', train_data[0].shape


def get_train_data():
  return train_data

def get_test_data():
  return test_data

def linear_acc(x1, x2, ys):
  x1 = x1.copy() * np.array([1, 0.05, 1, 1, 0.1, 0.25])
  x2 = x2.copy() * np.array([1, 0.05, 1, 1, 0.1, 0.25])
  cnt = 0
  for i, y in enumerate(ys):
    if y == 0.0 and np.sum(x1[i]) > np.sum(x2[i]):
      cnt += 1
    if y == 1.0 and np.sum(x1[i]) < np.sum(x2[i]):
      cnt += 1
  return cnt * 1. / len(ys)

def linear_rank(xs, name):
  return zip(*sorted([(sum(y), x) for y, x in zip((xs * np.array([1, 0.05, 1, 1, 0.1, 0.25])).tolist(), name)]))

if __name__ == '__main__':
  print '\n'.join(linear_rank(*load_model_data(token))[1])
