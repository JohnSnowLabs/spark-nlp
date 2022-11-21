---
layout: model
title: Financial Zero-shot NER
author: John Snow Labs
name: finner_roberta_zeroshot
date: 2022-09-02
tags: [en, finance, ner, zero, shot, zeroshot, licensed]
task: Named Entity Recognition
language: en
edition: Finance NLP 1.0.0
spark_version: 3.0
supported: true
recommended: true
annotator: ZeroShotNER
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This model is trained to carry out a Zero-Shot Named Entity Recognition (NER) approach, detecting any kind of entities with no training dataset, just tje pretrained RoBERTa embeddings (included in the model) and some examples.

## Predicted Entities



{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/finance/FINNER_ZEROSHOT/){:.button.button-orange}
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/finance/models/finner_roberta_zeroshot_en_1.0.0_3.2_1662113599526.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}

```python
documentAssembler = nlp.DocumentAssembler()\
  .setInputCol("text")\
  .setOutputCol("document")

sparktokenizer = nlp.Tokenizer()\
  .setInputCols("document")\
  .setOutputCol("token")

zero_shot_ner = finance.ZeroShotNerModel.pretrained("finner_roberta_zeroshot", "en", "finance/models")\
    .setInputCols(["document", "token"])\
    .setOutputCol("zero_shot_ner")\
    .setEntityDefinitions(
        {
            "DATE": ['When was the company acquisition?', 'When was the company purchase agreement?'],
            "ORG": ["Which company was acquired?"],
            "PRODUCT": ["Which product?"],
            "PROFIT_INCREASE": ["How much has the gross profit increased?"],
            "REVENUES_DECLINED": ["How much has the revenues declined?"],
            "OPERATING_LOSS_2020": ["Which was the operating loss in 2020"],
            "OPERATING_LOSS_2019": ["Which was the operating loss in 2019"]
        })

nerconverter = nlp.NerConverter()\
  .setInputCols(["document", "token", "zero_shot_ner"])\
  .setOutputCol("ner_chunk")

pipeline =  Pipeline(stages=[
  documentAssembler,
  sparktokenizer,
  zero_shot_ner,
  nerconverter,
    ]
)

sample_text = ["In March 2012, as part of a longer-term strategy, the Company acquired Vertro, Inc., which owned and operated the ALOT product portfolio.",
              "In February 2017, the Company entered into an asset purchase agreement with NetSeer, Inc.",
              "While our gross profit margin increased to 81.4% in 2020 from 63.1% in 2019, our revenues declined approximately 27% in 2020 as compared to 2019."
              "We reported an operating loss of approximately $8,048,581 million in 2020 as compared to an operating loss of approximately $7,738,193 million in 2019."]

p_model = pipeline.fit(spark.createDataFrame([[""]]).toDF("text"))

res = p_model.transform(spark.createDataFrame(sample_text, StringType()).toDF("text"))

res.select(F.explode(F.arrays_zip(res.ner_chunk.result, res.ner_chunk.begin, res.ner_chunk.end, res.ner_chunk.metadata)).alias("cols")) \
   .select(F.expr("cols['0']").alias("chunk"),
           F.expr("cols['3']['entity']").alias("ner_label"))\
   .filter("ner_label!='O'")\
   .show(truncate=False)
```

</div>

## Results

```bash
+------------------+-------------------+
|chunk             |ner_label          |
+------------------+-------------------+
|March 2012        |DATE               |
|Vertro            |ORG                |
|ALOT              |PRODUCT            |
|February 2017     |DATE               |
|NetSeer           |ORG                |
|81.4%             |PROFIT_INCREASE    |
|27%               |REVENUES_DECLINED  |
|$8,048,581 million|OPERATING_LOSS_2020|
|$7,738,193 million|OPERATING_LOSS_2019|
+------------------+-------------------+
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|finner_roberta_zeroshot|
|Type:|finance|
|Compatibility:|Finance NLP 1.0.0+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[document_question, document_context]|
|Output Labels:|[answer]|
|Language:|en|
|Size:|460.2 MB|
|Case sensitive:|true|
|Max sentence length:|512|

## References

Financial Roberta Embeddings
