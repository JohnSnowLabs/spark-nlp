import React from 'react';
import Tooltip from '../Tooltip';

const data = {
  medium:
    'Model or Pipeline which uses average amounts of data (usually few hundreds per class) with default word/sentence embeddings (bert etc).',
  small:
    'Model or Pipeline which uses low amounts of data (usually about 100 per class) and/or low-dimensional word/sentence embeddings (word2vec, etc).',
  fast: 'Used the fastest algorithms in this model/pipeline (speed over accuracy).',
  large:
    'Model or Pipeline which much more data and representativity in the classes than md, and/or based on bert or bigger transformers (t5, etc).',
  xlarge:
    'Model or Pipeline which much more data and representativity in the classes than lg, and/or based on bert or bigger transformers (t5, etc).',
  slim: 'Reduced version of the models, trained with less data, or reduced number of labels. Performance will be more or less the same given the use case. The model size and the inference speed will be same.',
  greedy:
    'Greedy strategies, similarly as for regex, mean that given NER chunks close to each other, it decides to merge them in 1 instead of returning them separately.',

  biobert:
    "Models were trained not with Word Embeddings, but with Bert Embeddings, more specifically with clinical Bert Embeddings (BioBert). Results may be better, but it requires including the BertEmbeddings component instead of WordEmbeddings. It's heavier but may bring better results.",
  enriched:
    'A version with more data and probably more entities than the original one (without the _enriched keyword).',
  augmented:
    'A version with more data and probably more entities than the original one (without the _augmented keyword).',
  wip: 'Work in progress models. Models that we update as we provide data.',
  modifier: 'Models used to extract modifiers within clinic entities.',
  cased: 'This model differentiates between lowercase and uppercase.',
  uncased: 'This model does not differentiate between lowercase and uppercase.',
  normalized:
    'In-house standardization and normalization of input and output labels.',
  base: 'Standard version of the model, used as a starting point to create bigger (large) and smaller (small, xsmall, etc) versions of the model.',
  xsmall:
    'Even smaller than small. Uses a very reduced size of embeddings to provide with better performance.',
  tuned:
    'Model finetuned by John Snow Labs on public and inhouse datasets, to provide better accuracy.',
  unidirectional: [
    'This model was trained to take into account the direction of the relation. ',
    <em>chunk1</em>,
    ' will always be the source of the relation, ',
    <em>chunk2</em>,
    ' the target.',
  ],
  bidirectional:
    'This model was not trained to take into consideration the direction of the relation, meaning that it can return relations from left to right or right to left indistinctly.',
};

// aliases
data.sm = data.small;
data.md = data.medium;
data.lg = data.large;
data.xl = data.xlarge;

export const addNamingConventions = (title) => {
  const parts = title.split(' ');
  return parts.reduce((acc, value, index) => {
    const cleanedValue = value.replace(/[^a-zA-Z]/g, '').toLowerCase();
    const content = data[cleanedValue];
    if (content) {
      acc.push(
        <Tooltip label={content} key={value}>
          <span style={{ borderBottom: '2px dotted' }}>{value}</span>
        </Tooltip>
      );
    } else {
      acc.push(value);
    }
    if (index < parts.length - 1) {
      acc.push(' ');
    }
    return acc;
  }, []);
};

const oldToNewProduct = {
  'Spark NLP': 'Spark NLP',
  'Spark NLP for Healthcare': 'Healthcare NLP',
  'Spark OCR': 'Visual NLP',
  'Spark NLP for Finance': 'Finance NLP',
  'Spark NLP for Legal': 'Legal NLP',
};

export const products = Object.values(oldToNewProduct);

export const productDisplayName = (edition) => {
  if (typeof edition === 'string') {
    for (const [key, value] of Object.entries(oldToNewProduct).reverse()) {
      if (edition.includes(key)) {
        return edition.replace(key, value);
      }
    }
  }
  return edition;
};
