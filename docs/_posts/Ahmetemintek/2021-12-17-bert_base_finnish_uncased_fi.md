---
layout: model
title: Finnish BERT Embeddings (Base Uncased)
author: Ahmetemintek
name: bert_base_finnish_uncased
date: 2021-12-17
tags: [open_source, embeddings, fi, bert]
task: Embeddings
language: fi
edition: Spark NLP 3.4.0
spark_version: 3.0
supported: false
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

A version of Google's BERT deep transfer learning model for Finnish. The model can be fine-tuned to achieve state-of-the-art results for various Finnish natural language processing tasks.

FinBERT features a custom 50,000 wordpiece vocabulary that has much better coverage of Finnish words than e.g. the previously released multilingual BERT models from Google.

FinBERT has been pre-trained for 1 million steps on over 3 billion tokens (24B characters) of Finnish text drawn from news, online discussion, and internet crawls. By contrast, Multilingual BERT was trained on Wikipedia texts, where the Finnish Wikipedia text is approximately 3% of the amount used to train FinBERT.

## Predicted Entities



{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/community.johnsnowlabs.com/Ahmetemintek/bert_base_finnish_uncased_fi_3.4.0_3.0_1639753349928.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
...
embeddings = BertEmbeddings.pretrained("bert_base_finnish_uncased", "fi") \
      .setInputCols("sentence", "token") \
      .setOutputCol("embeddings")

nlp_pipeline = Pipeline(stages=[document_assembler, sentence_detector, tokenizer, embeddings])
pipeline_model = nlp_pipeline.fit(spark.createDataFrame([[""]]).toDF("text"))
result = pipeline_model.transform(spark.createDataFrame([['Rakastan NLP: tä']], ["text"]))
```
```scala
...
val embeddings = BertEmbeddings.pretrained("bert_base_finnish_uncased", "fi")
      .setInputCols("sentence", "token")
      .setOutputCol("embeddings")
val pipeline = new Pipeline().setStages(Array(document_assembler, sentence_detector, tokenizer, embeddings))
val data = Seq("Rakastan NLP: tä").toDF("text")
val result = pipeline.fit(data).transform(data)
```
</div>

## Results

```bash
+--------------------+--------+
|          embeddings|   token|
+--------------------+--------+
|[0.75768787, 0.99...|Rakastan|
|[0.4912446, 0.388...|     NLP|
|[0.9057018, 0.253...|       :|
|[-0.6122347, 0.55...|      tä|
+--------------------+--------+
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|bert_base_finnish_uncased|
|Compatibility:|Spark NLP 3.4.0+|
|License:|Open Source|
|Edition:|Community|
|Input Labels:|[sentence, token]|
|Output Labels:|[bert]|
|Language:|fi|
|Size:|467.2 MB|
|Case sensitive:|true|