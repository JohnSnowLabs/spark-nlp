---
layout: model
title: Judgements Classifier (Agent)
author: John Snow Labs
name: legclf_bert_judgements_agent
date: 2022-08-30
tags: [en, legal, classification, judgements, actors, agents, court, decisions, licensed]
task: Text Classification
language: en
edition: Spark NLP for Legal 1.0.0
spark_version: 3.2
supported: true
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This is a Text Classification model, aimed to identify different entities in Court Decisions texts. More specifically, this model extracts the Agent (Actor or Legal Part). This model was inspired by [this](https://arxiv.org/pdf/2208.06178.pdf) paper, which uses a different approach (Named Entity Recognition).

The classes are listed below. Please check the [original paper](https://arxiv.org/pdf/2208.06178.pdf) for more information about them.

- APPLICANT
- COMMISSION/CHAMBER
- ECHR
- OTHER
- STATE
- THIRD PARTIES

## Predicted Entities

`APPLICANT`, `COMMISSION/CHAMBER`, `ECHR`, `OTHER`, `STATE`, `THIRD PARTIES`

{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/legal/LEGMULTICLF_LEDGAR/){:.button.button-orange}
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/legal/models/legclf_bert_judgements_agent_en_1.0.0_3.2_1661867874061.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
document_assembler = DocumentAssembler() \
    .setInputCol('text') \
    .setOutputCol('document')

tokenizer = Tokenizer()\
    .setInputCols(['document'])\
    .setOutputCol("token")

clf_model = LegalBertForSequenceClassification.pretrained("legclf_bert_judgements_agent", "en", "legal/models")\
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

text_list = ["""The applicant further noted that his placement in the home had already lasted more than eight years and that his hopes of leaving one day were futile , as the decision had to be approved by his guardian.""".lower(),
             """The Court observes that the situation was subsequently presented differently before the Riga Regional Court , the applicant having submitted , in the context of her appeal , a certificate prepared at her request by a psychologist on 16 December 2008 , that is , after the first - instance judgment . This document indicated that , while the child 's young age prevented her from expressing a preference as to her place of residence , an immediate separation from her mother was to be ruled out on account of the likelihood of psychological trauma ( see paragraph 22 above ).""".lower()
             ]
             
df = spark.createDataFrame(pd.DataFrame({"text" : text_list}))

result = model.transform(df)
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
|Compatibility:|Spark NLP for Legal 1.0.0+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[document, token]|
|Output Labels:|[class]|
|Language:|en|
|Size:|409.9 MB|
|Case sensitive:|false|
|Max sentence length:|512|

## References

https://arxiv.org/pdf/2208.06178.pdf

## Benchmarking

```bash
 				precision    recall  f1-score   support

         APPLICANT       0.90      0.90      0.90       238
COMMISSION/CHAMBER       0.90      0.95      0.93        20
              ECHR       0.93      0.96      0.94       870
             OTHER       0.94      0.91      0.93       940
             STATE       0.91      0.95      0.93       205
     THIRD PARTIES       0.96      0.85      0.90        26

          accuracy                           0.93      2299
         macro avg       0.92      0.92      0.92      2299
      weighted avg       0.93      0.93      0.93      2299
```