---
layout: model
title: Judgements Classification (Argument Type)
author: John Snow Labs
name: legclf_bert_judgements_argtype
date: 2022-09-07
tags: [en, legal, judgements, argument, echr, licensed]
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

This is a Text Classification model, aimed to identify different the different argument types in Court Decisions texts about Human Rights. This model was inspired by [this](https://arxiv.org/pdf/2208.06178.pdf) paper, which uses a different approach (Named Entity Recognition).

The classes are listed below. Please check the [original paper](https://arxiv.org/pdf/2208.06178.pdf) for more information about them.

- APPLICATION CASE
- DECISION ECHR
- LEGAL BASIS
- LEGITIMATE PURPOSE
- NECESSITY/PROPORTIONALITY
- NON CONTESTATION
- OTHER
- PRECEDENTS ECHR

## Predicted Entities

`APPLICATION CASE`, `DECISION ECHR`, `LEGAL BASIS`, `LEGITIMATE PURPOSE`, `NECESSITY/PROPORTIONALITY`, `NON CONTESTATION`, `OTHER`, `PRECEDENTS ECHR`

{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/finance/LEG_JUDGEMENTS_CLF/){:.button.button-orange}
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/legal/models/legclf_bert_judgements_argtype_en_1.0.0_3.2_1662562438186.zip){:.button.button-orange.button-orange-trans.arr.button-icon.hidden}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/legal/models/legclf_bert_judgements_argtype_en_1.0.0_3.2_1662562438186.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}

```python
text_list = ["""Indeed, given that the expectation of protection of private life may be reduced on account of the public functions exercised, the Court considers that, in order to ensure a fair balancing of the interests at stake, the domestic courts, in assessing the facts submitted for their examination, ought to have taken into account the potential impact of the Prince's status as Head of State, and to have attempted, in that context, to determine the parts of the impugned article that belonged to the strictly private domain and what fell within the public sphere.""",
             """Article 8 requires that the domestic authorities should strike a fair balance between the interests of the child and those of the parents, and that, in the balancing process, particular importance should be attached to the best interests of the child, which, depending on their nature and seriousness, may override those of the parents. In particular, a parent can not be entitled under Article 8 to have such measures taken as would harm the child's health and development ( see Sahin, cited above, § 66, and Sommerfeld, cited above, § 64 )."""]


text_list = [x.lower() for x in text_list]

text_list


document_assembler = nlp.DocumentAssembler() \
    .setInputCol('text') \
    .setOutputCol('document')

tokenizer = nlp.Tokenizer()\
    .setInputCols(['document'])\
    .setOutputCol("token")

clf_model = legal.BertForSequenceClassification.pretrained("legclf_bert_judgements_argtype", "en", "legal/models")\
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

result.select(F.explode(F.arrays_zip('document.result', 'class.result')).alias("cols"))\
               .select(F.expr("cols['0']").alias("document"),
                       F.expr("cols['1']").alias("class")).show(truncate = 60)
```

</div>

## Results

```bash
+------------------------------------------------------------+----------------+
|                                                    document|           class|
+------------------------------------------------------------+----------------+
|indeed, given that the expectation of protection of priva...|APPLICATION CASE|
|article 8 requires that the domestic authorities should s...| PRECEDENTS ECHR|
+------------------------------------------------------------+----------------+
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|legclf_bert_judgements_argtype|
|Type:|legal|
|Compatibility:|Legal NLP 1.0.0+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[document, token]|
|Output Labels:|[class]|
|Language:|en|
|Size:|410.0 MB|
|Case sensitive:|true|
|Max sentence length:|512|

## References

Basedf on https://arxiv.org/pdf/2208.06178.pdf with in-house postprocessing

## Benchmarking

```bash
                    label  precision    recall  f1-score   support
         APPLICATION_CASE       0.85      0.83      0.84       983
            DECISION_ECHR       0.82      0.86      0.84       103
              LEGAL_BASIS       0.61      0.50      0.55        40
       LEGITIMATE_PURPOSE       0.94      0.88      0.91        17
NECESSITY/PROPORTIONALITY       0.62      0.66      0.64       207
         NON_CONTESTATION       0.64      0.69      0.67        13
                    OTHER       0.97      0.97      0.97      2557
          PRECEDENTS_ECHR       0.80      0.85      0.83       262
                 accuracy         -         -       0.91      4182
                macro-avg       0.78      0.78      0.78      4182
             weighted-avg       0.91      0.91      0.91      4182
```