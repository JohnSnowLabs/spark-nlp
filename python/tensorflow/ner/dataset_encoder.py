import string
import random


class DatasetEncoder:
    # Each sentence must be array of tuple (word, tag)
    def __init__(self, word2id, embeddings, tag2id={'O': 0}):
        self.word2id = word2id
        self.char2id = {c:i + 1 for i, c in enumerate(string.printable)}
        self.tag2id = tag2id
        self.embeddings = embeddings
        
    def shuffle(self):
        random.shuffle(self.sentences)
    
    @staticmethod
    def normalize(word):
        return word.strip().lower()
    
    def get_char_indexes(self, word):
        result = []
        for c in word:
            char_id = self.char2id.get(c, len(self.char2id) - 1)
            result.append(char_id)

        return result
    
    def encode(self, sentences, output=False):
        for sentence in sentences:
            word_ids = []
            char_ids = []
            tag_ids = []
            words = []
            tags = []
            word_embeddings = []
            
            for (word, tag) in sentence:
                # Additional values
                normal_word = DatasetEncoder.normalize(word)
                tag_id = self.tag2id.get(tag, len(self.tag2id))
                self.tag2id[tag] = tag_id

                # Source texts
                words.append(word)
                tags.append(tag)
                              
                # Ids
                word_id = self.word2id.get(normal_word, 0)
                word_ids.append(word_id)
                char_ids.append(self.get_char_indexes(word))
                tag_ids.append(tag_id) 
                
                # Embeddings
                word_embeddings.append(self.embeddings[word_id])
            
            if len(sentence) > 0:
                yield {
                        "words": words,
                        "tags": tags,
                        "word_ids": word_ids,
                        "char_ids": char_ids,
                        "tag_ids": tag_ids,
                        "word_embeddings": word_embeddings
                    }
