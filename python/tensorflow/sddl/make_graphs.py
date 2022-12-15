import tensorflow.compat.v1 as tf
import tensorflow.keras as K

from arguments import parse_arguments

args = parse_arguments()

window_size = int(args.window_size)
epochs = int(args.epochs)
batch_size = int(args.batch_size)
dropout = float(args.dropout)
min_freq = int(args.min_freq)
max_features = int(args.max_features)
embedding_size = int(args.embedding_size)
lstm_gru_size = int(args.lstm_gru_size)
mlp_dense = int(args.mlp_dense)
mlp_dense_units = int(args.mlp_dense_units)
kernel_size = int(args.kernel_size)
filters = int(args.filters)
pool_size = int(args.pool_size)
hidden_dims = int(args.hidden_dims)
strides = int(args.strides)
                     
maxlen = 2 * window_size + 1

dropout = 0.0
max_features = 300

#CNN model

graph_path = "/models/sent/graphs"

def save_graph(graph, model_id):
    
    tf.io.write_graph(
        graph, 
        "/tmp/{}".format(model_id), 
        "{}/{}.pb".format(graph_path, model_id), 
        False)
    print("Graph exported to {}/{}.pb".format(graph_path, model_id))
    
def make_cnn_model():
    global max_features, embedding_size, maxlen, kernel_size, dropout
    global hidden_dims
    
    tf.reset_default_graph()
    
    with tf.Session() as session:
        
        #Input
        input_layer = tf.placeholder(
            tf.float32, shape=[None, maxlen], name='inputs')

        #Targets
        targets = tf.placeholder(
            tf.float32, shape=[None, 1], name='targets')

        #Learning rate
        learning_rate = tf.placeholder_with_default(
            tf.constant(0.001, dtype=tf.float32),
            shape=[], name='learning_rate')

        #Class weight (not used)
        tf.placeholder_with_default(
            tf.ones(shape=1),
            shape=[1],
            name='class_weights')

        #Dropout (not used)
        tf.placeholder_with_default(
            tf.constant(0.0, dtype=tf.float32),
            shape=[],
            name='dropout')

        embeddings_layer = K.layers.Embedding(
            max_features, embedding_size, input_length=maxlen)(input_layer)
        
        dropout_layer1 = tf.nn.dropout(
                embeddings_layer * 1.,
                rate=tf.minimum(1.0, tf.minimum(1.0, dropout)))
    
        conv_layer = K.layers.Conv1D(filters,
                          kernel_size,
                          padding='valid',
                          activation='relu',
                          strides=1)(dropout_layer1)
        
        pooling_layer = K.layers.GlobalMaxPooling1D()(conv_layer)
        hidden_layer = K.layers.Dense(hidden_dims)(pooling_layer)

        dropout_layer2 = tf.nn.dropout(
                hidden_layer * 1.,
                rate=tf.minimum(1.0, tf.minimum(1.0, dropout)))
        
        relu_layer = K.layers.Activation('relu')(dropout_layer2)
        output_layer = K.layers.Dense(
            units=1,
            activation=None)(relu_layer)
                    
        outputs = tf.math.sigmoid(output_layer, name="outputs")
            
        #Loss function
        loss = tf.reduce_mean(
            K.losses.binary_crossentropy(
                targets,
                outputs),
            name='loss')

        #Optimizer
        tf.train.AdamOptimizer(
            learning_rate=learning_rate,
            name='optimizer').minimize(loss)

        #Accuracy per trial
        correct_prediction = tf.equal(
            tf.greater(tf.reduce_sum(output_layer, 1), 0.5),
            tf.greater(tf.reduce_sum(targets, 1), 0.5),
            name='correct_prediction')

        #Overall accuracy/by batch
        tf.reduce_mean(
            tf.cast(correct_prediction, tf.float32),
            name='accuracy')

        #Model predictions
        tf.argmax(
            output_layer,
            axis=1,
            name='predictions')
            
        #TF variables initialization
        init = tf.global_variables_initializer()

        tf.train.Saver()

        return session.graph
   
        
cnn_graph = make_cnn_model()
save_graph(cnn_graph, "cnn")

# import re

# t = """
# I am using the Keras functional API to create a neural net that does the 
# following:(repeat until true).
# """

# PUNCT = '[\(\)\u0093\u0094`“”\"›〈⟨〈<‹»«‘’–\'``'']*'
# EOS = '([\.:?!;])'
# pattern = r'([\.:?!;])(\s+' + PUNCT + '|' + PUNCT + '\s+' + '|' + '[\s\n]+)'

# eos_positions = [(m.start()) for m in re.finditer(pattern, t)]

# prev_pos = 0
# for pos in eos_positions:
#     print(t[prev_pos:pos] + "<eos>" + t[pos] + "</eos>", end="")
#     prev_pos = pos + 1

# print(t[prev_pos:])