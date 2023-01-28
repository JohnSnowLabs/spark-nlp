---
layout: model
title: Judgements Classification (agent)
author: John Snow Labs
name: legclf_bert_judgements_agent
date: 2022-09-07
tags: [en, legal, judgements, agent, echr, licensed]
task: Text Classification
language: en
edition: Legal NLP 1.0.0
spark_version: 3.0
supported: true
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This is a Text Classification model, aimed to identify different the different argument types in Court Decisions texts about Human Rights. This model was inspired by [this](https://arxiv.org/pdf/2208.06178.pdf) paper, which uses a different approach (Named Entity Recognition). The model classifies the claims by the type of Agent (Party) involved (it's the Court talking, the applicant, ...).

The classes are listed below. Please check the [original paper](https://arxiv.org/pdf/2208.06178.pdf) for more information about them.

## Predicted Entities

`APPLICANT`, `COMMISSION/CHAMBER`, `ECHR`, `OTHER`, `STATE`, `THIRD_PARTIES`

{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/finance/LEG_JUDGEMENTS_CLF/){:.button.button-orange}
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/legal/models/legclf_bert_judgements_agent_en_1.0.0_3.2_1662560852536.zip){:.button.button-orange.button-orange-trans.arr.button-icon.hidden}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/legal/models/legclf_bert_judgements_agent_en_1.0.0_3.2_1662560852536.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}

```python
text_list = ["""The applicant further noted that his placement in the home had already lasted more than eight years and that his hopes of leaving one day were futile , as the decision had to be approved by his guardian.""".lower(),
             """The Court observes that the situation was subsequently presented differently before the Riga Regional Court , the applicant having submitted , in the context of her appeal , a certificate prepared at her request by a psychologist on 16 December 2008 , that is , after the first - instance judgment . This document indicated that , while the child 's young age prevented her from expressing a preference as to her place of residence , an immediate separation from her mother was to be ruled out on account of the likelihood of psychological trauma ( see paragraph 22 above ).""".lower()
             ]
             
# Test classifier in Spark NLP pipeline
document_assembler = nlp.DocumentAssembler() \
    .setInputCol('text') \
    .setOutputCol('document')

tokenizer = nlp.Tokenizer()\
    .setInputCols(['document'])\
    .setOutputCol("token")

clf_model = legal.BertForSequenceClassification.pretrained("legclf_bert_judgements_agent", "en", "legal/models")\
    .setInputCols(['document','token'])\
    .setOutputCol("class")\
    .setCaseSensitive(True)\
    .setMaxSentenceLength(512)

clf_pipeline = Pipeline(stages=[
    document_assembler, 
    tokenizer,
    clf_model   
])

# Generating example
empty_df = spark.createDataFrame([['']]).toDF("text")

model = clf_pipeline.fit(empty_df)

light_model = LightPipeline(model)

import pandas as pd

df = spark.createDataFrame(pd.DataFrame({"text" : text_list}))

result = model.transform(df)

result = result.select(F.explode(F.arrays_zip('document.result', 'class.result')).alias("cols"))\
               .select(F.expr("cols['0']").alias("document"),
                       F.expr("cols['1']").alias("class")).show(truncate = 60)
```

</div>

## Results

```bash
+------------------------------------------------------------+---------+
|                                                    document|    class|
+------------------------------------------------------------+---------+
|the applicant further noted that his placement in the hom...|APPLICANT|
|the court observes that the situation was subsequently pr...|     ECHR|
+------------------------------------------------------------+---------+
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|legclf_bert_judgements_agent|
|Type:|legal|
|Compatibility:|Legal NLP 1.0.0+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[document, token]|
|Output Labels:|[class]|
|Language:|en|
|Size:|409.9 MB|
|Case sensitive:|false|
|Max sentence length:|512|

## References

Basedf on https://arxiv.org/pdf/2208.06178.pdf with in-house postprocessing

## Benchmarking

```bash
             label  precision    recall  f1-score   support
         APPLICANT       0.91      0.89      0.90       238
COMMISSION/CHAMBER       0.80      1.00      0.89        20
              ECHR       0.92      0.96      0.94       870
             OTHER       0.95      0.90      0.93       940
             STATE       0.91      0.94      0.92       205
     THIRD_PARTIES       0.96      0.92      0.94        26
          accuracy         -         -       0.93      2299
         macro-avg       0.91      0.94      0.92      2299
      weighted-avg       0.93      0.93      0.93      2299
```
