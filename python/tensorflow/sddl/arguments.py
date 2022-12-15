# -*- coding: utf-8 -*-
import argparse

def parse_arguments():
    parser = argparse.ArgumentParser()
    
    # parser.add_argument("mode", choices=["train", "test", "tag", "extract"])
    # parser.add_argument("--training-file",
    #                     help="Defines training data set")
    # parser.add_argument("--test-file",
    #                     help="Defines test data set")
    # parser.add_argument("--input-file",
    #                     help="Defines input file to be tagged")
    parser.add_argument("--epochs", default=5,
                        help="Defines number of training epochs")
    # parser.add_argument(
    #     "--architecture",
    #     default="cnn",
    #     help="Neural network architectures, supported: cnn, lstm, bi-lstm, gru, bi-gru, mlp")
    parser.add_argument("--window-size", default=5,
                        help="Defines number of window size (char-ngram)")
    parser.add_argument("--batch-size", default=32,
                        help="Defines number of batch_size")
    parser.add_argument("--dropout", default=0.2,
                        help="Defines number dropout")
    parser.add_argument(
        "--min-freq",
        default=100,
        help="Defines the min. freq. a char must appear in data")
    parser.add_argument("--max-features", default=200,
                        help="Defines number of features for Embeddings layer")
    parser.add_argument("--embedding-size", default=128,
                        help="Defines Embeddings size")
    parser.add_argument("--kernel-size", default=8,
                        help="Defines Kernel size of CNN")
    parser.add_argument("--filters", default=6,
                        help="Defines number of filters of CNN")
    parser.add_argument("--pool-size", default=8,
                        help="Defines pool size of CNN")
    parser.add_argument("--hidden-dims", default=250,
                        help="Defines number of hidden dims")
    parser.add_argument("--strides", default=1,
                        help="Defines numer of strides for CNN")
    parser.add_argument("--lstm_gru_size", default=256,
                        help="Defines size of LSTM/GRU layer")
    parser.add_argument("--mlp-dense", default=6,
                        help="Defines number of dense layers for mlp")
    parser.add_argument("--mlp-dense-units", default=16,
                        help="Defines number of dense units for mlp")
    parser.add_argument("--model-filename", default='best_model.hdf5',
                        help="Defines model filename")
    parser.add_argument("--vocab-filename", default='vocab.dump',
                        help="Defines vocab filename")
    parser.add_argument("--eos-marker", default='</eos>',
                        help="Defines end-of-sentence marker used for tagging")
    return parser.parse_args()
        