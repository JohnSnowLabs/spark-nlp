---
layout: model
title: Financial NER for German Financial Statements
author: John Snow Labs
name: finner_financial_entity_value
date: 2023-03-24
tags: [en, ner, licensed, finance, de]
task: Named Entity Recognition
language: de
edition: Finance NLP 1.0.0
spark_version: 3.0
supported: true
annotator: FinanceNerModel
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This is a German NER model trained on German Financial Statements, aimed to extract the following entities from the financial documents.

## Predicted Entities

`FINANCIAL_ENTITY`, `FINANCIAL_VALUE`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/finance/models/finner_financial_entity_value_de_1.0.0_3.0_1679699846660.zip){:.button.button-orange}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/finance/models/finner_financial_entity_value_de_1.0.0_3.0_1679699846660.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}

```python
documentAssembler = nlp.DocumentAssembler()\
        .setInputCol("text")\
        .setOutputCol("document")
        
sentenceDetector = nlp.SentenceDetectorDLModel.pretrained("sentence_detector_dl","xx")\
        .setInputCols(["document"])\
        .setOutputCol("sentence")

tokenizer = nlp.Tokenizer()\
        .setInputCols(["sentence"])\
        .setOutputCol("token")

embeddings = nlp.BertEmbeddings.pretrained("bert_sentence_embeddings_financial","de") \
    .setInputCols("sentence", "token") \
    .setOutputCol("embeddings")

ner_model= finance.NerModel.pretrained("finner_financial_entity_value_de", "de", "finance/models")\
  .setInputCols(["sentence", "token", "embeddings"])\
  .setOutputCol("ner")

ner_converter = nlp.NerConverter()\
        .setInputCols(["sentence","token","ner"])\
        .setOutputCol("ner_chunk")

pipeline =  nlp.Pipeline(stages=[
    documentAssembler,
    sentenceDetector,
    tokenizer,
    embeddings,
    ner_model,
    ner_converter
    ]
)

import pandas as pd

p_model = pipeline.fit(spark.createDataFrame(pd.DataFrame({'text': ['']})))


text = 'Die Kapitalstruktur wird im Wesentlichen durch eine weitere Reduzierung der langfristigen Bankverbindlichkeiten um 3.000 TEUR auf 0 TEUR , einer Erhöhung der Rückstellungen um 3.397 TEUR auf 31.717 TEUR sowie die Erhöhung des Eigenkapitals um 1.771 TEUR auf 110.668 TEUR beeinflusst .'


res = p_model.transform(spark.createDataFrame([[text]]).toDF("text"))

result_df = res.select(F.explode(F.arrays_zip(res.token.result,res.ner.result, res.ner.metadata)).alias("cols"))\
                          .select(F.expr("cols['0']").alias("token"),
                                       F.expr("cols['1']").alias("label"),
                                       F.expr("cols['2']['confidence']").alias("confidence"))

result_df.show(50, truncate=100)
```

</div>

## Results

```bash
+---------------------+------------------+
|                token|             label|
+---------------------+------------------+
|                  Die|                 O|
|      Kapitalstruktur|                 O|
|                 wird|                 O|
|                   im|                 O|
|         Wesentlichen|                 O|
|                durch|                 O|
|                 eine|                 O|
|              weitere|                 O|
|          Reduzierung|                 O|
|                  der|                 O|
|        langfristigen|B-FINANCIAL_ENTITY|
|Bankverbindlichkeiten|I-FINANCIAL_ENTITY|
|                   um|                 O|
|                3.000|                 O|
|                 TEUR|                 O|
|                  auf|                 O|
|                    0| B-FINANCIAL_VALUE|
|                 TEUR|                 O|
|                    ,|                 O|
|                einer|                 O|
|             Erhöhung|                 O|
|                  der|                 O|
|       Rückstellungen|B-FINANCIAL_ENTITY|
|                   um|                 O|
|                3.397|                 O|
|                 TEUR|                 O|
|                  auf|                 O|
|               31.717| B-FINANCIAL_VALUE|
|                 TEUR|                 O|
|                sowie|                 O|
|                  die|                 O|
|             Erhöhung|                 O|
|                  des|                 O|
|        Eigenkapitals|B-FINANCIAL_ENTITY|
|                   um|                 O|
|                1.771|                 O|
|                 TEUR|                 O|
|                  auf|                 O|
|              110.668| B-FINANCIAL_VALUE|
|                 TEUR|                 O|
|          beeinflusst|                 O|
|                    .|                 O|
+---------------------+------------------+
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|finner_financial_entity_value|
|Compatibility:|Finance NLP 1.0.0+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[sentence, token, embeddings]|
|Output Labels:|[ner]|
|Language:|de|
|Size:|1.1 MB|

## References

Train dataset available [here](https://huggingface.co/datasets/fabianrausch/financial-entities-values-augmented)

## Benchmarking

```bash
 label               precision   recall      f1-score   support     
 B-FINANCIAL_ENTITY  0.8947      0.9444      0.9189     18 
 B-FINANCIAL_VALUE   1.0000      0.8750      0.9333     16 
 I-FINANCIAL_ENTITY  0.8000      0.6154      0.6957     13 
 micro-avg           0.9070      0.8298      0.8667     47 
 macro-avg           0.8982      0.8116      0.8493     47 
 weighted-avg        0.9044      0.8298      0.8621     47 
```
