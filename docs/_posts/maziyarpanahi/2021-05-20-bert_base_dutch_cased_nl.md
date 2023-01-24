---
layout: model
title: BERTje A Dutch BERT model
author: John Snow Labs
name: bert_base_dutch_cased
date: 2021-05-20
tags: [open_source, embeddings, bert, dutch, nl]
task: Embeddings
language: nl
edition: Spark NLP 3.1.0
spark_version: 3.0
supported: true
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

BERTje is a Dutch pre-trained BERT model developed at the University of Groningen.

For details, check out our paper on [arXiv](https://arxiv.org/abs/1912.09582), the code on [Github](https://github.com/wietsedv/bertje) and related work on [Semantic Scholar](https://www.semanticscholar.org/paper/BERTje%3A-A-Dutch-BERT-Model-Vries-Cranenburgh/a4d5e425cac0bf84c86c0c9f720b6339d6288ffa).
The paper and Github page mention fine-tuned models that are available [here](https://huggingface.co/wietsedv).

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/bert_base_dutch_cased_nl_3.1.0_3.0_1621500934814.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/bert_base_dutch_cased_nl_3.1.0_3.0_1621500934814.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
embeddings = BertEmbeddings.pretrained("bert_base_dutch_cased", "nl") \
      .setInputCols("sentence", "token") \
      .setOutputCol("embeddings")
nlp_pipeline = Pipeline(stages=[document_assembler, sentence_detector, tokenizer, embeddings])
```
```scala
val embeddings = BertEmbeddings.pretrained("bert_base_dutch_cased", "nl")
      .setInputCols("sentence", "token")
      .setOutputCol("embeddings")
val pipeline = new Pipeline().setStages(Array(document_assembler, sentence_detector, tokenizer, embeddings))
```


{:.nlu-block}
```python
import nlu
nlu.load("nl.embed.bert").predict("""Put your text here.""")
```

</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|bert_base_dutch_cased|
|Compatibility:|Spark NLP 3.1.0+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[token, sentence]|
|Output Labels:|[embeddings]|
|Language:|nl|
|Case sensitive:|true|

## Data Source

https://huggingface.co/dbmdz/bert-base-german-cased

## Benchmarking

```bash
The arXiv paper lists benchmarks. Here are a couple of comparisons between BERTje, multilingual BERT, BERT-NL, and RobBERT that were done after writing the paper. Unlike some other comparisons, the fine-tuning procedures for these benchmarks are identical for each pre-trained model. You may be able to achieve higher scores for individual models by optimizing fine-tuning procedures.
More experimental results will be added to this page when they are finished. Technical details about how a fine-tuned these models will be published later as well as downloadable fine-tuned checkpoints.
All of the tested models are *base* sized (12) layers with cased tokenization.
Headers in the tables below link to original data sources. Scores link to the model pages that correspond to that specific fine-tuned model. These tables will be updated when more simple fine-tuned models are made available.
### Named Entity Recognition
| Model                                                                        | [CoNLL-2002](https://www.clips.uantwerpen.be/conll2002/ner/)                                  | [SoNaR-1](https://ivdnt.org/downloads/taalmaterialen/tstc-sonar-corpus)                   | spaCy UD LassySmall                                                                             |
| ---------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------- |
| **BERTje**                                                                   | [**90.24**](https://huggingface.co/wietsedv/bert-base-dutch-cased-finetuned-conll2002-ner)    | [**84.93**](https://huggingface.co/wietsedv/bert-base-dutch-cased-finetuned-sonar-ner)    | [86.10](https://huggingface.co/wietsedv/bert-base-dutch-cased-finetuned-udlassy-ner)            |
| [mBERT](https://github.com/google-research/bert/blob/master/multilingual.md) | [88.61](https://huggingface.co/wietsedv/bert-base-multilingual-cased-finetuned-conll2002-ner) | [84.19](https://huggingface.co/wietsedv/bert-base-multilingual-cased-finetuned-sonar-ner) | [**86.77**](https://huggingface.co/wietsedv/bert-base-multilingual-cased-finetuned-udlassy-ner) |
| [BERT-NL](http://textdata.nl)                                                | 85.05                                                                                         | 80.45                                                                                     | 81.62                                                                                           |
| [RobBERT](https://github.com/iPieter/RobBERT)                                | 84.72                                                                                         | 81.98                                                                                     | 79.84                                                                                           |
### Part-of-speech tagging
| Model                                                                        | [UDv2.5 LassySmall](https://universaldependencies.org/treebanks/nl_lassysmall/index.html) |
| ---------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------- |
| **BERTje**                                                                   | **96.48**                                                                                 |
| [mBERT](https://github.com/google-research/bert/blob/master/multilingual.md) | 96.20                                                                                     |
| [BERT-NL](http://textdata.nl)                                                | 96.10                                                                                     |
| [RobBERT](https://github.com/iPieter/RobBERT)                                | 95.91                                                                                     |

```
