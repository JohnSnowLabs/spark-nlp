---
layout: model
title: Russian BertForSequenceClassification Base Cased model (from cointegrated)
author: John Snow Labs
name: bert_sequence_classifier_ru_base_cased_nli_threeway
date: 2023-03-16
tags: [ru, open_source, bert, sequence_classification, ner, tensorflow]
task: Named Entity Recognition
language: ru
edition: Spark NLP 4.3.1
spark_version: 3.0
supported: true
engine: tensorflow
annotator: BertForSequenceClassification
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained BertForSequenceClassification model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP. `rubert-base-cased-nli-threeway` is a Russian model originally trained by `cointegrated`.

## Predicted Entities

`neutral`, `contradiction`, `entailment`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/bert_sequence_classifier_ru_base_cased_nli_threeway_ru_4.3.1_3.0_1678984048161.zip){:.button.button-orange}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/bert_sequence_classifier_ru_base_cased_nli_threeway_ru_4.3.1_3.0_1678984048161.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
documentAssembler = DocumentAssembler() \
    .setInputCol("text") \
    .setOutputCol("document")

tokenizer = Tokenizer() \
    .setInputCols("document") \
    .setOutputCol("token")

sequenceClassifier = BertForSequenceClassification.pretrained("bert_sequence_classifier_ru_base_cased_nli_threeway","ru") \
    .setInputCols(["document", "token"]) \
    .setOutputCol("class")

pipeline = Pipeline(stages=[documentAssembler, tokenizer, sequenceClassifier])

data = spark.createDataFrame([["PUT YOUR STRING HERE"]]).toDF("text")

result = pipeline.fit(data).transform(data)
```
```scala
val documentAssembler = new DocumentAssembler()
    .setInputCol("text")
    .setOutputCol("document")

val tokenizer = new Tokenizer()
    .setInputCols("document")
    .setOutputCol("token")

val sequenceClassifier = BertForSequenceClassification.pretrained("bert_sequence_classifier_ru_base_cased_nli_threeway","ru")
    .setInputCols(Array("document", "token"))
    .setOutputCol("ner")

val pipeline = new Pipeline().setStages(Array(documentAssembler, tokenizer, sequenceClassifier))

val data = Seq("PUT YOUR STRING HERE").toDS.toDF("text")

val result = pipeline.fit(data).transform(data)
```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|bert_sequence_classifier_ru_base_cased_nli_threeway|
|Compatibility:|Spark NLP 4.3.1+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[document, token]|
|Output Labels:|[ner]|
|Language:|ru|
|Size:|667.1 MB|
|Case sensitive:|true|
|Max sentence length:|128|

## References

- https://huggingface.co/cointegrated/rubert-base-cased-nli-threeway
- https://github.com/felipessalvatore/NLI_datasets
- https://github.com/sheng-z/JOCI
- https://cims.nyu.edu/~sbowman/multinli/
- https://aclanthology.org/I17-1011/
- http://www.lrec-conf.org/proceedings/lrec2014/pdf/363_Paper.pdf
- https://nlp.stanford.edu/projects/snli/
- https://github.com/facebookresearch/anli
- https://github.com/easonnie/combine-FEVER-NSMN/blob/master/other_resources/nli_fever.md
- https://github.com/facebookresearch/Imppres
- https://cs.brown.edu/people/epavlick/papers/ans.pdf
- https://people.ict.usc.edu/~gordon/copa.html
- https://aclanthology.org/I17-1100
- https://allenai.org/data/scitail
- https://github.com/felipessalvatore/NLI_datasets
- https://github.com/verypluming/HELP
- https://github.com/atticusg/MoNLI
- https://russiansuperglue.com/ru/tasks/task_info/TERRa