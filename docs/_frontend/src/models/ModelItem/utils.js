import React from 'react';
import Tooltip from '../Tooltip';

const data = {
  medium:
    'Used the average size Word/Sentence Embeddings in this model/pipeline.',
  small:
    'Used the smallest size Word/Sentence Embeddings in this model/pipeline.',
  fast: 'Used the fastest algorithms in this model/pipeline (speed over accuracy).',
  large:
    'Trained on more data, which means they can generalize further and can provide better results in some cases. The model size and the inference speed will be same.',

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
};

export const addNamingConventions = (title) => {
  const parts = title.split(' ');
  return parts.reduce((acc, value, index) => {
    const cleanedValue = value.replace(/[^a-zA-Z]/g, '').toLowerCase();
    const content = data[cleanedValue];
    if (content) {
      acc.push(
        <Tooltip label={content}>
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

export const products = {
  'Spark NLP': 'Spark NLP',
  'Spark NLP for Healthcare': 'Healthcare NLP',
  'Spark OCR': 'Visual NLP',
  'Spark NLP for Finance': 'Finance NLP',
  'Spark NLP for Legal': 'Legal NLP',
};

export const productDisplayName = (edition) => {
  for (const [key, value] of Object.entries(products).reverse()) {
    if (edition.includes(key)) {
      return edition.replace(key, value);
    }
  }
};
