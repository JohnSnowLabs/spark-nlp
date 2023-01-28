---
layout: model
title: Named Entity Recognition for Bengali (GloVe 840B 300d)
author: John Snow Labs
name: ner_jifs_glove_840B_300d
date: 2021-01-27
task: Named Entity Recognition
language: bn
edition: Spark NLP 2.7.0
spark_version: 2.4
tags: [bn, ner, open_source]
supported: true
annotator: NerDLModel
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This model annotates named entities in a text, that can be used to find features such as names of people, places, and organizations. The model does not read words directly but instead reads word embeddings, which represent words as points such that more semantically similar words are closer together.

This model uses the pre-trained `glove_840B_300` embeddings model from `WordEmbeddings` annotator as an input, so be sure to use the same embeddings in the pipeline.

## Predicted Entities

`PER`, `LOC`, `ORG`, `OBJ`, `O`

{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/public/NER_EN/){:.button.button-orange}
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/NER.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/ner_jifs_glove_840B_300d_bn_2.7.0_2.4_1611770574503.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/ner_jifs_glove_840B_300d_bn_2.7.0_2.4_1611770574503.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

document_assembler = DocumentAssembler() \
    .setInputCol("text") \
    .setOutputCol("document")

sentence_detector = SentenceDetector()\
    .setInputCols(["document"])\
    .setOutputCol("sentence")

tokenizer = Tokenizer()\
    .setInputCols("sentence")\
    .setOutputCol("token")

embeddings = WordEmbeddingsModel.pretrained("glove_840B_300", "xx")\
.setInputCols("document", "token") \
.setOutputCol("embeddings")

ner = NerDLModel.pretrained("ner_jifs_glove_840B_300d", "bn") \
.setInputCols(["document", "token", "embeddings"]) \
.setOutputCol("ner")

pipeline = Pipeline(stages=[document_assembler, sentence_detector, tokenizer, embeddings, ner])
example = spark.createDataFrame([["৯০ এর দশকের শুরুর দিকে বৃহৎ আকারে মার্কিন যুক্তরাষ্ট্রে এর প্রয়োগের প্রক্রিয়া শুরু হয়'"]], ["text"])
result = pipeline.fit(example).transform(example)
```

```scala

val document_assembler = DocumentAssembler()
        .setInputCol("text")
        .setOutputCol("document")
        
val sentence_detector = SentenceDetector()\
    .setInputCols(["document"])\
    .setOutputCol("sentence")

val tokenizer = Tokenizer()\
    .setInputCols("sentence")\
    .setOutputCol("token")

val embeddings = WordEmbeddingsModel.pretrained("glove_840B_300", "xx")
.setInputCols(Array("document", "token"))
.setOutputCol("embeddings")

val ner = NerDLModel.pretrained("ner_jifs_glove_840B_300d", "bn")
.setInputCols(Array("document", "token", "embeddings"))
.setOutputCol("ner")

val pipeline = new Pipeline().setStages(Array(document_assembler, sentence_detector, tokenizer, embeddings, ner))
val data = Seq("৯০ এর দশকের শুরুর দিকে বৃহৎ আকারে মার্কিন যুক্তরাষ্ট্রে এর প্রয়োগের প্রক্রিয়া শুরু হয়").toDF("text")
val result = pipeline.fit(data).transform(data)
```

{:.nlu-block}
```python
import nlu

text = ["৯০ এর দশকের শুরুর দিকে বৃহৎ আকারে মার্কিন যুক্তরাষ্ট্রে এর প্রয়োগের প্রক্রিয়া শুরু হয়"]
ner_df = nlu.load('bn.ner').predict(text, output_level='token')
ner_df
```

</div>

## Results

```bash
+-------------+-----+
|token        |ner  |
+-------------+-----+
|৯০           |O    |
|এর           |O    |
|দশকের        |O    |
|শুরুর        |O    |
|দিকে         |O    |
|বৃহৎ         |O    |
|আকারে        |O    |
|মার্কিন      |B-LOC|
|যুক্তরাষ্ট্রে|I-LOC|
|এর           |O    |
|প্রয়োগের    |O    |
|প্রক্রিয়া   |O    |
|শুরু         |O    |
|হয়          |O    |
|'            |O    |
+-------------+-----+
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|ner_jifs_glove_840B_300d|
|Type:|ner|
|Compatibility:|Spark NLP 2.7.0+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[sentence, token, embeddings]|
|Output Labels:|[ner]|
|Language:|bn|

## Data Source

The model was trained on the [Bengali NER](https://github.com/MISabic/NER-Bangla-Dataset) data set introduced in the Journal of Intelligent & Fuzzy Systems.

Reference:

- Karim, Redwanul & Islam, M. A. & Simanto, Sazid & Chowdhury, Saif & Roy, Kalyan & Neon, Adnan & Hasan, Md & Firoze, Adnan & Rahman, Mohammad. (2019). A step towards information extraction: Named entity recognition in Bangla using deep learning. Journal of Intelligent & Fuzzy Systems. 37. 1-13. 10.3233/JIFS-179349.

## Benchmarking

```bash
|              | precision | recall | f1-score | support |
|--------------|-----------|--------|----------|---------|
| B-LOC        | 0.81      | 0.72   | 0.76     | 2005    |
| B-OBJ        | 0.66      | 0.08   | 0.13     | 573     |
| B-ORG        | 0.67      | 0.31   | 0.42     | 853     |
| B-PER        | 0.76      | 0.76   | 0.76     | 4035    |
| I-LOC        | 0.64      | 0.52   | 0.58     | 357     |
| I-OBJ        | 0.00      | 0.00   | 0.00     | 57      |
| I-ORG        | 0.65      | 0.37   | 0.47     | 516     |
| I-PER        | 0.76      | 0.73   | 0.74     | 1223    |
| O            | 0.93      | 0.97   | 0.95     | 39499   |
| accuracy     |           |        | 0.90     | 49118   |
| macro avg    | 0.65      | 0.49   | 0.54     | 49118   |
| weighted avg | 0.89      | 0.90   | 0.89     | 49118   |
```
