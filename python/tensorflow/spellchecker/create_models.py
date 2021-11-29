
import os
from rnn_lm import RNNLM


def create_graph(hunits, num_layers, classes, vocab_size):
    model = RNNLM(batch_size=24,
                  num_epochs=5,
                  check_point_step=5000,
                  num_layers=num_layers,
                  num_hidden_units=hunits,
                  max_gradient_norm=5.0,
                  max_num_classes=classes,
                  max_word_ids=classes,
                  vocab_size=vocab_size,
                  initial_learning_rate=.7,
                  final_learning_rate=0.0005)

    # Persist graph
    model.persist_graph('nlm_%d_%d_%d_%d.pb' % (hunits, num_layers, classes, vocab_size))

    return model


if __name__ == '__main__':
    create_graph(300, 2, 2000, 56650)
