import shutil
import numpy as np
import plyvel
import os.path
import sys
sys.path.append('../')
from bert.modeling import *
from bert.tokenization import *
import json
import os.path
import numpy as np


class TokenEmbeddings:
    def __init__(self, piece, is_word_start, vector):
        self.piece = piece
        self.is_word_start = is_word_start
        self.vector = vector
    
    @staticmethod
    def create_sentence(pieces, is_word_starts, embeddings):
        # Array of TokenEmbeddings
        return [TokenEmbeddings(piece, is_start, vector)
            for (piece, is_start, vector) in zip(pieces, is_word_starts, embeddings)]
    
    def __str__(self):
        return 'TokenEmbeddings({}, {}, [{}])'.format(self.piece, self.is_word_start, np.shape(self.vector))

    def __repr__(self):
        return self.__str__()


class EmbeddingsDbResolver:
    
    @staticmethod
    def get_index_name(prefix, dim):
        return prefix + '-' + str(dim)
    
    @staticmethod
    def read_from_file(glove_file, dim, index_file = 'embeddings_index', 
                       lowercase=False, clear_if_exists = False):
        
        full_index_file = EmbeddingsDbResolver.get_index_name(index_file, dim)
        try:
            resolver = None

            index_existed = os.path.exists(full_index_file) and not clear_if_exists
            resolver = EmbeddingsDbResolver(dim, index_file, lowercase, clear_if_exists)

            if not index_existed:
                resolver.read_glove(glove_file)

            return resolver
        except:
            if resolver and resolver.db:
                resolver.close()
            
            raise()
            
    def read_glove(self, glove_file):
        portion = 500000
        print('reading file: ', glove_file)
        wb = None
        with open(glove_file, encoding='utf-8') as f:
            for num, line in enumerate(f):
                items = line.split(' ')
                word = items[0]
                vector = [float(x) for x in items[1:]]
                if num % portion == portion - 1:
                    print('read lines: {}'.format(num))
                    wb.write()
                    wb = None
                
                if not wb:
                    wb = self.db.write_batch()

                self.add_vector(word, vector, wb)
            if wb:
                wb.write()
        
    def __init__(self, dim, index_file = 'embeddings_index', lowercase = False, clear_if_exists=False):        
        full_index_file = EmbeddingsDbResolver.get_index_name(index_file, dim)
        
        if clear_if_exists and os.path.exists(full_index_file):
            shutil.rmtree(db_index)
        
        dummy_added = False
        self.db = plyvel.DB(full_index_file, create_if_missing=True)
        self.add_vector("__oov__", [0.] * dim)
        self.lowercase = lowercase
        
    def get_embeddings(self, word):
        word = word.strip()
        if self.lowercase:
            word = word.lower()
            
        result = self.db.get(word.encode()) or self.db.get('__oov__'.encode())
        return np.frombuffer(result)
    
    def resolve_sentence(self, sentence):
        """
        sentence - array of words
        """
        embeddings =  list([self.get_embeddings(word) for word in sentence])
        is_word_start = [True] * len(sentence)
        
        return TokenEmbeddings.create_sentence(sentence, is_word_start, embeddings)

            
    def add_vector(self, word, vector, wb = None):
        array = np.array(vector)
        if wb:
            wb.put(word.encode(), array.tobytes())
        else:
            self.db.put(word.encode(), array.tobytes())
    
    def close(self):
        self.db.close()
        

class BertEmbeddingsResolver:
    
    def __init__(self, model_folder, max_length = 256, lowercase = True):
        
        # 1. Create tokenizer
        self.max_length = max_length
        vocab_file = os.path.join(model_folder, 'vocab.txt')
        self.tokenizer = FullTokenizer(vocab_file, do_lower_case = lowercase)
        
        # 2. Read Config
        config_file = os.path.join(model_folder, 'bert_config.json')        
        self.config = BertConfig.from_json_file(config_file)
        
        # 3. Create Model
        self.session = tf.Session()
        self.token_ids_op = tf.placeholder(tf.int32, shape=(None, max_length), name='token_ids')
        self.model = BertModel(config = self.config, 
                          is_training = False, 
                          input_ids = self.token_ids_op, 
                          use_one_hot_embeddings = False)
        
        # 4. Restore Trained Model
        self.saver = tf.train.Saver()
        ckpt_file = os.path.join(model_folder, 'bert_model.ckpt')
        self.saver.restore(self.session, ckpt_file)
        
        hidden_layers = self.config.num_hidden_layers
        self.embeddings_op = tf.get_default_graph().get_tensor_by_name(
            "bert/encoder/Reshape_{}:0".format(hidden_layers + 1))
        
    def tokenize_sentence(self, tokens, add_service_tokens = True):
        result = []
        is_word_start = []
        for token in tokens:
            pieces = self.tokenizer.tokenize(token)
            result.extend(pieces)
            starts = [False] * len(pieces)
            starts[0] = True
            is_word_start.extend(starts)

        if add_service_tokens:
            if len(result) > self.max_length - 2:
                result = result[:self.max_length -2]
                is_word_start = is_word_start[:self.max_length -2]
            
            result = ['[CLS]'] + result + ['[SEP]']
            is_word_start = [False] + is_word_start + [False]
        else:
            if len(result) > self.max_length:
                result = result[:self.max_length]
                is_word_start = is_word_start[:self.max_length]
        
        return (result, is_word_start)

    def resolve_sentences(self, sentences):
        batch_is_word_start = []
        batch_token_ids = []
        batch_tokens = []
        
        for sentence in sentences:
            tokens, is_word_start = self.tokenize_sentence(sentence)
            token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
            to_input = np.pad(token_ids, [(0, self.max_length - len(token_ids))], mode='constant')
            batch_token_ids.append(to_input)
            batch_tokens.append(tokens)
            batch_is_word_start.append(is_word_start)

        embeddings = self.session.run(self.embeddings_op, feed_dict = {self.token_ids_op: batch_token_ids})
        
        result = []
        for i in range(len(sentences)):
            tokens = batch_tokens[i]
            is_word_start = batch_is_word_start[i]
            item_embeddings = embeddings[i, :len(tokens), :]

            resolved = TokenEmbeddings.create_sentence(tokens, is_word_start, item_embeddings)
            result.append(resolved)
        
        return result

    
    def resolve_sentence(self, sentence):
        tokens, is_word_start = self.tokenize_sentence(sentence)
        
        token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        to_input = np.pad(token_ids, [(0, self.max_length - len(token_ids))], mode='constant')
        to_input = to_input.reshape((1, self.max_length))

        embeddings = self.session.run(self.embeddings_op, feed_dict = {self.token_ids_op: to_input})
        embeddings = np.squeeze(embeddings)
        embeddings = embeddings[:len(token_ids), :]

        return TokenEmbeddings.create_sentence(tokens, is_word_start, embeddings)
        