import mxnet as mx
import numpy as np
import logging
logging.getLogger().setLevel(logging.DEBUG)  #设置logger输出级别

train_label_iter=mx.io.CSVIter(data_csv='train_labels.csv', data_shape=(2,), batch_size=10)

for train_label_batch in train_label_iter:
    train_label_ndarray = train_label_batch.data[0].asnumpy()
#    print len(train_label_batch.data), type(train_label_batch.data)
#    print len(train_label_ndarray), type(train_label_ndarray), train_label_ndarray.shape
#    print train_label_ndarray
    for i in range(len(train_label_ndarray)):
        print i, train_label_ndarray[i]
#    for train_label in np.nditer(train_label_ndarray):
#        print train_label.shape
#        print train_label

train_pic_iter = mx.image.ImageIter(batch_size=10, data_shape=(3,1156,866),path_imglist='train_imglist.lst', path_root='train')
train_pic_batch=train_pic_iter.next()
#print train_pic_batch
#print train_pic_batch.data,  type(train_pic_batch.data), len(train_pic_batch.data)
#print train_pic_batch.data[0], type(train_pic_batch.data[0])
#print train_pic_batch.label[0]
#for i in range(len(train_pic_batch.data[0].asnumpy())):
#    print i, train_pic_batch.data[0][i].asnumpy()

data = mx.symbol.Variable('data')

net1 = mx.symbol.FullyConnected(data=data, name='fc1', num_hidden=10)

