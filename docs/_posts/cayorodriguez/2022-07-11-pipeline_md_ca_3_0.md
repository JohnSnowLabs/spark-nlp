---
layout: model
title: Basic General Purpose Pipeline for Catalan
author: cayorodriguez
name: pipeline_md
date: 2022-07-11
tags: [ca, open_source]
task: [Named Entity Recognition, Sentence Detection, Embeddings, Stop Words Removal, Part of Speech Tagging, Lemmatization, Chunk Mapping, Pipeline Public]
language: ca
edition: Spark NLP 3.4.4
spark_version: 3.0
supported: false
recommended: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Model for Catalan language processing based on models by Barcelona SuperComputing Center and the AINA project (Generalitat de Catalunya), following POS and tokenization guidelines from ANCORA Universal Dependencies corpus.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Public/1.SparkNLP_Basics.ipynb){:.button.button-orange.button-orange-trans.arr.button-icon}
[Download](https://s3.amazonaws.com/community.johnsnowlabs.com/cayorodriguez/pipeline_md_ca_3.4.4_3.0_1657533114488.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://community.johnsnowlabs.com/cayorodriguez/pipeline_md_ca_3.4.4_3.0_1657533114488.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
pipeline = PretrainedPipeline("pipeline_md", "ca", "@cayorodriguez")

result = pipeline.annotate("El català ja és a SparkNLP.")
```

</div>

## Results

```bash
{'chunk': ['El català ja', 'SparkNLP', 'és'],
 'entities': ['SparkNLP'],
 'lemma': ['el', 'català', 'ja', 'ser', 'a', 'sparknlp', '.'],
 'document': ['El català ja es a SparkNLP.'],
 'pos': ['DET', 'NOUN', 'ADV', 'AUX', 'ADP', 'PROPN', 'PUNCT'],
 'sentence_embeddings': ['El català ja és a SparkNLP.'],
 'cleanTokens': ['català', 'SparkNLP', '.'],
 'token': ['El', 'català', 'ja', 'és', 'a', 'SparkNLP', '.'],
 'ner': ['O', 'O', 'O', 'O', 'O', 'B-ORG', 'O'],
 'embeddings': ['El', 'català', 'ja', 'és', 'a', 'SparkNLP', '.'],
 'form': ['el', 'català', 'ja', 'és', 'a', 'sparknlp', '.'],
 'sentence': ['El català ja és a SparkNLP.']}
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|pipeline_md|
|Type:|pipeline|
|Compatibility:|Spark NLP 3.4.4+|
|License:|Open Source|
|Edition:|Community|
|Language:|ca|
|Size:|756.1 MB|

## Included Models

- DocumentAssembler
- SentenceDetectorDLModel
- TokenizerModel
- NormalizerModel
- StopWordsCleaner
- RoBertaEmbeddings
- SentenceEmbeddings
- EmbeddingsFinisher
- LemmatizerModel
- PerceptronModel
- RoBertaForTokenClassification
- NerConverter
- Chunker
