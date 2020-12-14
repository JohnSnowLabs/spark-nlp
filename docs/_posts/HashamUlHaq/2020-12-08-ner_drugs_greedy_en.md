---
layout: model
title: Detect Drugs - generalized single entity (ner_drugs_greedy)
author: John Snow Labs
name: ner_drugs_greedy
date: 2020-12-08
tags: [ner, licensed, en, clinical]
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This is a single entity model that generalises all posology concepts into one and finds longest available chunks of drugs. It is trained using `embeddings_clinical` so please use the same embeddings in the pipeline.

## Predicted Entities

\``DRUG`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/ner_drugs_greedy_en_2.6.4_2.4_1607417409084.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use

Use as part of an nlp pipeline with the following stages: DocumentAssembler, SentenceDetector, Tokenizer, WordEmbeddingsModel, NerDLModel. Add the NerConverter to the end of the pipeline to convert entity tokens into full entity chunks.

<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPython.html %}
```python
clinical_ner = NerDLModel.pretrained("ner_drugs_greedy", "en", "clinical/models") \
  .setInputCols(["sentence", "token", "embeddings"]) \
  .setOutputCol("ner")
nlp_pipeline = Pipeline(stages=[
    document_assembler, 
    sentence_detector,
    tokenizer,
    word_embeddings,
    clinical_ner,
    ner_converter])
model = nlpPipeline.fit(spark.createDataFrame([[""]]).toDF("text"))
data = "DOSAGE AND ADMINISTRATION The initial dosage of hydrocortisone tablets may vary from 20 mg to 240 mg of hydrocortisone per day depending on the specific disease entity being treated."
results = model.transform(spark.createDataFrame([[data]]).toDF("text"))
```

</div>

## Results

```bash
+-----------------------------------+------------+
| chunk                             | ner_label  |
+-----------------------------------+------------+
| hydrocortisone tablets            | DRUG       |
| 20 mg to 240 mg of hydrocortisone | DRUG       |
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|ner_drugs_greedy|
|Type:|ner|
|Compatibility:|Spark NLP 2.6.4+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[sentence, token, embeddings]|
|Output Labels:|[ner]|
|Language:|en|
|Dependencies:|embeddings_clinical|

## Benchmarking

```bash
label	 tp	 fp	 fn	 prec	 rec	 f1
I-DRUG	 37858	 4166	 3338	 0.90086615	 0.91897273	 0.9098294
B-DRUG	 29926	 2006	 1756	 0.937179	 0.9445742	 0.9408621
tp: 67784 fp: 6172 fn: 5094 labels: 2
Macro-average	 prec: 0.91902256, rec: 0.9317734, f1: 0.92535406
Micro-average	 prec: 0.916545, rec: 0.93010235, f1: 0.9232739
```