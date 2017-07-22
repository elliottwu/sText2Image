import numpy as np
import pickle
import pdb

attr_file = 'list_attr_celeba.txt'
cal_percent = False

aoi_idxs = [1,7,8,14,15,16,17,19,20,21,22,23,24,27,28,30,32,37]
#aoi_idxs = range(1,41)

attr_sum = np.zeros(len(aoi_idxs))

with open(attr_file, 'r') as f:
    lines = f.readlines()
    ttl_no = int(lines.pop(0))
    results = np.zeros((ttl_no, len(aoi_idxs)))

    headers = lines.pop(0).split()
    aoi = [headers[i-1] for i in aoi_idxs]

    assert len(lines) == ttl_no

    for i in range(ttl_no):
        if (i%1000 == 0):
            print("%i/%i" %(i,ttl_no))

        line = lines[i]
        attrs_str = line.split()
        im_name = attrs_str.pop(0)
        attrs_filtered = [int(attrs_str[i-1]) for i in aoi_idxs]
        attrs_int = np.array(attrs_filtered).astype(np.int)
        attr_sum += attrs_int
        results[i,:] = attrs_int

pickle.dump(results, open('imAttrs.pkl','wb'))

if (cal_percent):
    attr_percent = (1 + attr_sum/ttl_no) / 2
    with open('attr_percent.txt', 'w') as f_percent:
        for i in range(len(aoi_idxs)):
            f_percent.write('%-20s %.4f\n' % (aoi[i], attr_percent[i]))
