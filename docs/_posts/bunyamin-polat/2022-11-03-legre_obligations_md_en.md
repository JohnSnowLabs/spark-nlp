---
layout: model
title: Legal Relation Extraction (Obligations, md, Unidirectional)
author: John Snow Labs
name: legre_obligations_md
date: 2022-11-03
tags: [en, legal, licensed, obligation, re]
task: Relation Extraction
language: en
edition: Legal NLP 1.0.0
spark_version: 3.0
supported: true
annotator: RelationExtractionDLModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description
IMPORTANT: Don't run this model on the whole legal agreement. Instead:
- Split by paragraphs. You can use [notebook 1](https://github.com/JohnSnowLabs/spark-nlp-workshop/tree/master/tutorials/Certification_Trainings_JSL) in Finance or Legal as inspiration;
- Use the `legclf_cuad_obligations_clause` Text Classifier to select only these paragraphs; 

We call "obligation" to any sentence in the text stating that a Party (OBLIGATION_SUBJECT) must do (OBLIGATION_ACITON) something (OBLIGATION_OBJECT) to other Party (OBLIGATION_INDIRECT_OBJECT). This model extracts relationships, connecting all of those parts of the sentence (subject with action, action with object, etc).

This model requires `legner_obligations` as an NER in the pipeline.It's a `md` model with Unidirectional Relations, meaning that the model retrieves in chunk1 the left side of the relation (source), and in chunk2 the right side (target).

This is a Deep Learning model, meaning only semantics are taking into account, not grammatical structures. If you want to parse the relations using a grammatical dependency tree, please feel free to use [this other model](https://nlp.johnsnowlabs.com/2022/08/24/legpipe_obligations_en.html)

## Predicted Entities

`is_obliged_to`, `is_obliged_with`, `is_obliged_object`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/legal/models/legre_obligations_md_en_1.0.0_3.0_1667474780413.zip){:.button.button-orange.button-orange-trans.arr.button-icon.hidden}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/legal/models/legre_obligations_md_en_1.0.0_3.0_1667474780413.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}

```python
document_assembler = nlp.DocumentAssembler()\
    .setInputCol("text")\
    .setOutputCol("document")

tokenizer = nlp.Tokenizer()\
    .setInputCols("document")\
    .setOutputCol("token")

ner_model = legal.BertForTokenClassification.pretrained("legner_obligations", "en", "legal/models")\
    .setInputCols("token", "document")\
    .setOutputCol("ner")\
    .setCaseSensitive(True)

ner_converter = nlp.NerConverter()\
    .setInputCols(["document","token","ner"])\
    .setOutputCol("ner_chunk")

re_model = legal.RelationExtractionDLModel()\
    .pretrained("legre_obligations_md", "en", "legal/models")\
    .setPredictionThreshold(0.5)\
    .setInputCols(["ner_chunk", "document"])\
    .setOutputCol("relations")

pipeline = Pipeline(stages=[
        document_assembler, 
        tokenizer,
        ner_model,
        ner_converter,
        re_model
])

empty_df = spark.createDataFrame([[""]]).toDF("text")

model = pipeline.fit(empty_df)

text="""Licensee agrees to reasonably cooperate with Licensor in achieving registration of the Licensed Mark."""

data = spark.createDataFrame([[text]]).toDF("text")

result = model.transform(data)


```

</div>

## Results

```bash
| relation          | entity1                    | entity1_begin | entity1_end | chunk1                         | entity2                    | entity2_begin | entity2_end | chunk2                                         | confidence |
|-------------------|----------------------------|---------------|-------------|--------------------------------|----------------------------|---------------|-------------|------------------------------------------------|------------|
| is_obliged_to     | OBLIGATION_ACTION          | 9             | 38          | agrees to reasonably cooperate | OBLIGATION_SUBJECT         | 0             | 7           | Licensee                                       | 0.91654503 |
| is_obliged_with   | OBLIGATION_SUBJECT         | 0             | 7           | Licensee                       | OBLIGATION_INDIRECT_OBJECT | 45            | 52          | Licensor                                       | 0.803172   |
| is_obliged_to     | OBLIGATION_SUBJECT         | 0             | 7           | Licensee                       | OBLIGATION                 | 54            | 99          | in achieving registration of the Licensed Mark | 0.7439706  |
| is_obliged_object | OBLIGATION_ACTION          | 9             | 38          | agrees to reasonably cooperate | OBLIGATION_INDIRECT_OBJECT | 45            | 52          | Licensor                                       | 0.96132916 |
| is_obliged_object | OBLIGATION_ACTION          | 9             | 38          | agrees to reasonably cooperate | OBLIGATION                 | 54            | 99          | in achieving registration of the Licensed Mark | 0.9174475  |
| is_obliged_to     | OBLIGATION_INDIRECT_OBJECT | 45            | 52          | Licensor                       | OBLIGATION                 | 54            | 99          | in achieving registration of the Licensed Mark | 0.9091029  |
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|legre_obligations_md|
|Compatibility:|Legal NLP 1.0.0+|
|License:|Licensed|
|Edition:|Official|
|Language:|en|
|Size:|402.3 MB|

## References

Manual annotations on CUAD dataset

## Benchmarking

```bash
| Relation          | Recall | Precision | F1    | Support |
|-------------------|--------|-----------|-------|---------|
| is_obliged_object | 0.989  | 0.994     | 0.992 | 177     |
| is_obliged_to     | 0.995  | 1.000     | 0.998 | 202     |
| is_obliged_with   | 1.000  | 0.961     | 0.980 | 49      |
| Avg.              | 0.996  | 0.989     | 0.992 | -       |
| Weighted-Avg.     | 0.996  | 0.996     | 0.996 | -       |
```
