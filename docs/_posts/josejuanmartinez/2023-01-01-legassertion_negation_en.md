---
layout: model
title: Legal Assertion Status (Negation)
author: John Snow Labs
name: legassertion_negation
date: 2023-01-01
tags: [negation, en, licensed]
task: Assertion Status
language: en
edition: Legal NLP 1.0.0
spark_version: 3.0
supported: true
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This is a Legal Negation model, aimed to identify if an NER entity is mentioned in the context to be negated or not.

## Predicted Entities

`positive`, `negative`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/legal/models/legassertion_negation_en_1.0.0_3.0_1672578547085.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
import pyspark.sql.functions as F

document_assembler = nlp.DocumentAssembler()\
    .setInputCol("text")\
    .setOutputCol("document")

sentence_detector = nlp.SentenceDetector() \
    .setInputCols(["document"]) \
    .setOutputCol("sentence")

tokenizer = nlp.Tokenizer()\
    .setInputCols(["sentence"])\
    .setOutputCol("token")

embeddings = nlp.BertEmbeddings.pretrained("bert_embeddings_sec_bert_base","en") \
    .setInputCols(["sentence", "token"]) \
    .setOutputCol("embeddings")

ner = legal.NerModel.pretrained("legner_orgs_prods_alias","en","legal/models")\
    .setInputCols(["sentence", "token", "embeddings"])\
    .setOutputCol("ner")

ner_converter = nlp.NerConverter() \
    .setInputCols(["sentence", "token", "ner"]) \
    .setOutputCol("ner_chunk")

legassertion = legal.AssertionDLModel.pretrained("legassertion_negation", "en", "legal/models")\
    .setInputCols(["sentence", "ner_chunk", "embeddings"])\
    .setOutputCol("leglabel")

pipe = nlp.Pipeline(stages = [ document_assembler, sentence_detector, tokenizer, embeddings, ner, ner_converter, legassertion])

text = "Gradio INC will not be entering into a joint agreement with Hugging Face, Inc."

sdf = spark.createDataFrame([[text]]).toDF("text")
res = pipe.fit(sdf).transform(sdf)

res.select(F.explode(F.arrays_zip(res.ner_chunk.result, 
                                  res.leglabel.result)).alias("cols"))\
                  .select(F.expr("cols['0']").alias("ner_chunk"),
                          F.expr("cols['1']").alias("assertion")).show(200, truncate=100)

```

</div>

## Results

```bash
+-----------------+---------+
|        ner_chunk|assertion|
+-----------------+---------+
|       Gradio INC| negative|
|Hugging Face, Inc| positive|
+-----------------+---------+
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|legassertion_negation|
|Compatibility:|Legal NLP 1.0.0+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[document, chunk, embeddings]|
|Output Labels:|[assertion]|
|Language:|en|
|Size:|2.2 MB|

## References

In-house annotated legal sentences

## Benchmarking

```bash
label	 tp	 fp	 fn	 prec	 rec	 f1
negative	 26	 0	 1	 1.0	 0.962963	 0.9811321
positive	 38	 1	 0	 0.974359	 1.0	 0.987013
Macro-average 641 1 1 0.9871795 0.9814815 0.9843222
Micro-average 0.9846154 0.9846154 0.9846154
```