# train assertion status on i2b2 dataset
from __future__ import print_function
from pyspark.sql import SparkSession
from assertion_model import AssertionModel
from data_access import I2b2Dataset


spark = SparkSession.builder \
    .appName("i2b2 tf bilstm") \
    .config("spark.driver.memory","4G") \
    .master("local[2]") \
    .getOrCreate()

trainset = I2b2Dataset('../../i2b2_train.csv', spark, extra_path='../../extra.csv')
testset = I2b2Dataset('../../i2b2_test.csv', spark)

#trainset = MockDataset(3072)
#testset = MockDataset(1024)

# don't need spark from now on
spark.stop()

print('Datasets read....')

# Parameters
learning_rate = 0.082
dropout = 0.25
batch_size = 128


for do in [0.005, 0.01, 0.02, 0.1, 0.15, 0.20, 0.25]:
    print('do: %f' % do)
    for nh in [32]:
        print('nh: %d' % nh)
        # instantiate model
        model = AssertionModel(trainset.max_seq_len, feat_size=210, n_classes=6, device='/gpu:0')

        # Network Parameters
        model.add_bidirectional_lstm(n_hidden=nh)
        model.train(trainset=trainset, testset=testset, learning_rate=0.0152, batch_size=batch_size, epochs=80)
