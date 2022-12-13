---
layout: model
title: Legal NER - License Grant Clauses (Md, Lighter version)
author: John Snow Labs
name: legner_grants_md
date: 2022-12-01
tags: [en, licensed]
task: Named Entity Recognition
language: en
edition: Legal NLP 1.0.0
spark_version: 3.0
supported: true
annotator: LegalClassifierDLModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This model aims to detect License grants / permissions in agreements, provided by a Subject (PERMISSION_SUBJECT) to a Recipient (PERMISSION_INDIRECT_OBJECT). THe permission itself is in PERMISSION tag.

This is a `md` (medium version) of the classifier, trained with more data and being more resistent to false positives.

This is also different from other permission models in that this only is lighter, non-transformer based.

## Predicted Entities

`PERMISSION`, `PERMISSION_SUBJECT`, `PERMISSION_OBJECT`, `PERMISSION_INDIRECT_OBJECT`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/legal/models/legner_grants_md_en_1.0.0_3.0_1669893713541.zip){:.button.button-orange.button-orange-trans.arr.button-icon.hidden}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/legal/models/legner_grants_md_en_1.0.0_3.0_1669893713541.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

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

embeddings = nlp.RoBertaEmbeddings.pretrained("roberta_embeddings_legal_roberta_base","en") \
    .setInputCols(["sentence", "token"]) \
    .setOutputCol("embeddings")

ner_model = legal.NerModel.pretrained('legner_grants_md', 'en', 'legal/models')\
        .setInputCols(["sentence", "token", "embeddings"])\
        .setOutputCol("ner")

ner_converter = nlp.NerConverter()\
        .setInputCols(["sentence","token","ner"])\
        .setOutputCol("ner_chunk")

nlpPipeline = Pipeline(stages=[documentAssembler,sentenceDetector,tokenizer,embeddings,ner_model,ner_converter])

import pandas as pd

p_model = pipeline.fit(spark.createDataFrame(pd.DataFrame({'text': ['']})))

text = """Fox grants to Licensee a limited, exclusive (except as otherwise may be provided in this Agreement), 
non-transferable (except as permitted in Paragraph 17(d)) right and license"""

res = p_model.transform(spark.createDataFrame([[text]]).toDF("text"))

from pyspark.sql import functions as F

res.select(F.explode(F.arrays_zip('token.result', 'label.result')).alias("cols")) \
               .select(F.expr("cols['0']").alias("token"),
                       F.expr("cols['1']").alias("ner_label"))\
               .show(20, truncate=100)
```

</div>

## Results

```bash
+----------------+----------------------------+
|           token|                   ner_label|
+----------------+----------------------------+
|             Fox|        B-PERMISSION_SUBJECT|
|          grants|                           O|
|              to|                           O|
|        Licensee|B-PERMISSION_INDIRECT_OBJECT|
|               a|                           O|
|         limited|                B-PERMISSION|
|               ,|                I-PERMISSION|
|       exclusive|                I-PERMISSION|
|               (|                I-PERMISSION|
|          except|                I-PERMISSION|
|              as|                I-PERMISSION|
|       otherwise|                I-PERMISSION|
|             may|                I-PERMISSION|
|              be|                I-PERMISSION|
|        provided|                I-PERMISSION|
|              in|                I-PERMISSION|
|            this|                I-PERMISSION|
|       Agreement|                I-PERMISSION|
|              ),|                I-PERMISSION|
|non-transferable|                I-PERMISSION|
+----------------+----------------------------+
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|legner_grants_md|
|Compatibility:|Legal NLP 1.0.0+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[sentence, token, embeddings]|
|Output Labels:|[ner]|
|Language:|en|
|Size:|16.1 MB|

## References

Manual annotations on CUAD dataset

## Benchmarking

```bash
label	 tp	 fp	 fn	 prec	 rec	 f1
I-PERMISSION	 111	 28	 37	 0.79856116	 0.75	 0.7735192
B-PERMISSION	 12	 3	 2	 0.8	 0.85714287	 0.82758623
B-PERMISSION_INDIRECT_OBJECT	 10	 1	 5	 0.90909094	 0.6666667	 0.7692308
B-PERMISSION_SUBJECT	 9	 1	 5	 0.9	 0.64285713	 0.74999994
Macro-average 142 33 52 0.68153036 0.5833334 0.72862015
Micro-average 142 33 52 0.81142855 0.73195875 0.76964766
```