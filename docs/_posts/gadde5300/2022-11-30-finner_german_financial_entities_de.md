---
layout: model
title: Financial NER for German Financial Statements
author: John Snow Labs
name: finner_german_financial_entities
date: 2022-11-30
tags: [licensed, de, ner, finance]
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

This is a German NER model trained on German Financial Statements, aimed to extract the following entities from the documents.

## Predicted Entities

`financial_entity`, `financial_value`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/finance/models/finner_german_financial_entities_de_1.0.0_3.0_1669806210718.zip){:.button.button-orange.button-orange-trans.arr.button-icon.hidden}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/finance/models/finner_german_financial_entities_de_1.0.0_3.0_1669806210718.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

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

embeddings = nlp.BertEmbeddings.pretrained("bert_embeddings_german_financial_statements_bert","de") \
    .setInputCols("sentence", "token") \
    .setOutputCol("embeddings")

tokenClassifier = finance.NerModel.pretrained("finner_german_financial_entities", "de", "finance/models")\
  .setInputCols(["sentence", "token", "embeddings"])\
  .setOutputCol("ner")

ner_converter = nlp.NerConverter()\
        .setInputCols(["sentence","token","ner"])\
        .setOutputCol("ner_chunk")

pipeline =  Pipeline(stages=[
  documentAssembler,
    sentenceDetector,
      tokenizer,
    embeddings,
  tokenClassifier,
    ner_converter
    ]
)

import pandas as pd

p_model = pipeline.fit(spark.createDataFrame(pd.DataFrame({'text': ['']})))


text = 'Die Steuern vom Einkommen und Etrag in Hoehe von TEUR 11.621 (Vorjahr TEUR 8.915) betreffen das Umlaufvermoegen'

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
+---------------+------------------+----------+
|          token|             label|confidence|
+---------------+------------------+----------+
|            Die|                 O|    0.9998|
|        Steuern|B-financial_entity|    0.9999|
|            vom|I-financial_entity|       1.0|
|      Einkommen|I-financial_entity|       1.0|
|            und|I-financial_entity|    0.9999|
|          Etrag|I-financial_entity|       1.0|
|             in|                 O|    0.9998|
|          Hoehe|                 O|       1.0|
|            von|                 O|       1.0|
|           TEUR|                 O|       1.0|
|         11.621| B-financial_value|       1.0|
|              (|                 O|       1.0|
|        Vorjahr|                 O|       1.0|
|           TEUR|                 O|       1.0|
|          8.915|                 O|       1.0|
|              )|                 O|       1.0|
|      betreffen|                 O|       1.0|
|            das|                 O|    0.9999|
|Umlaufvermoegen|B-financial_entity|    0.9999|
+---------------+------------------+----------+
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|finner_german_financial_entities|
|Compatibility:|Finance NLP 1.0.0+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[sentence, token, embeddings]|
|Output Labels:|[ner]|
|Language:|de|
|Size:|16.3 MB|

## References

https://huggingface.co/datasets/fabianrausch/financial-entities-values-augmented and in house JSL corrections and data augmentation.

## Benchmarking

```bash
             label   precision    recall  f1-score   support
B-financial_entity     0.9923    0.9983    0.9953      1813
 B-financial_value     1.0000    0.9920    0.9960      1369
I-financial_entity     0.9962    0.9998    0.9980      4148
                 O     0.9998    0.9989    0.9994     16197
          accuracy          -         -    0.9986     23527
         macro-avg     0.9971    0.9972    0.9971     23527
      weighted-avg     0.9986    0.9986    0.9986     23527
```
