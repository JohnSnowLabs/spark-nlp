---
layout: model
title: Legal Zero-shot Relation Extraction
author: John Snow Labs
name: legre_zero_shot
date: 2022-08-22
tags: [en, legal, re, zero, shot, zero_shot, licensed]
task: Relation Extraction
language: en
edition: Legal NLP 1.0.0
spark_version: 3.0
supported: true
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This is a Zero-shot Relation Extraction Model, meaning that it does not require any training data, just few examples of of the relations types you are looking for, to output a proper result.

Make sure you keep the proper syntax of the relations you want to extract. For example:

```
re_model.setRelationalCategories({
    "GRANTS_TO": ["{OBLIGATION_SUBJECT} grants {OBLIGATION_INDIRECT_OBJECT}"],
    "GRANTS": ["{OBLIGATION_SUBJECT} grants {OBLIGATION_ACTION}"]
})
```


- The keys of the dictionary are the name of the relations (`GRANTS_TO`, `GRANTS`)
- The values are list of sentences with similar examples of the relation
- The values in brackets are the NER labels extracted by an NER component before

## Predicted Entities



{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/legal/models/legre_zero_shot_en_1.0.0_3.2_1661181212397.zip){:.button.button-orange.button-orange-trans.arr.button-icon.hidden}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/legal/models/legre_zero_shot_en_1.0.0_3.2_1661181212397.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

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

tokenClassifier = legal.BertForTokenClassifier.pretrained('legner_obligations','en', 'legal/models')\
  .setInputCols("token", "document")\
  .setOutputCol("ner")\
  .setCaseSensitive(True)

ner_converter = nlp.NerConverter()\
    .setInputCols(["document", "token", "ner"])\
    .setOutputCol("ner_chunk")

re_model = legal.ZeroShotRelationExtractionModel.pretrained("legre_zero_shot", "en", "legal/models")\
    .setInputCols(["ner_chunk", "sentence"]) \
    .setOutputCol("relations")

# Remember it's 2 curly brackets instead of one if you are using Spark NLP < 4.0
re_model.setRelationalCategories({
    "GRANTS_TO": ["{OBLIGATION_SUBJECT} grants {OBLIGATION_INDIRECT_OBJECT}"],
    "GRANTS": ["{OBLIGATION_SUBJECT} grants {OBLIGATION_ACTION}"]
})

pipeline = sparknlp.base.Pipeline() \
    .setStages([document_assembler,  
                sparktokenizer,
                tokenClassifier, 
                ner_converter,
                re_model
               ])
               
# create Spark DF

sample_text = """Fox grants to Licensee a limited, exclusive right and license"""

data = spark.createDataFrame([[sample_text]]).toDF("text")
model = pipeline.fit(data)
results = model.transform(data)

# ner output
results.selectExpr("explode(ner_chunk) as ner").show(truncate=False)

# relations output
results.selectExpr("explode(relations) as relation").show(truncate=False)

```

</div>

## Results

```bash
+----------------------------------------------------------------------------------------------------------------------------+
|ner                                                                                                                         |
+----------------------------------------------------------------------------------------------------------------------------+
|[chunk, 0, 2, Fox, [entity -> OBLIGATION_SUBJECT, sentence -> 0, chunk -> 0, confidence -> 0.6905101], []]                  |
|[chunk, 4, 9, grants, [entity -> OBLIGATION_ACTION, sentence -> 0, chunk -> 1, confidence -> 0.7512371], []]                |
|[chunk, 14, 21, Licensee, [entity -> OBLIGATION_INDIRECT_OBJECT, sentence -> 0, chunk -> 2, confidence -> 0.8294538], []]   |
|[chunk, 23, 31, a limited, [entity -> OBLIGATION, sentence -> 0, chunk -> 3, confidence -> 0.7429814], []]                  |
|[chunk, 34, 60, exclusive right and license, [entity -> OBLIGATION, sentence -> 0, chunk -> 4, confidence -> 0.9236847], []]|
+----------------------------------------------------------------------------------------------------------------------------+

+-------------+
|relation     |
+-------------+
|[category, 0, 91, GRANTS, [entity1_begin -> 0, relation -> GRANTS, hypothesis -> Fox grants grants, confidence -> 0.7592092, nli_prediction -> entail, entity1 -> OBLIGATION_SUBJECT, syntactic_distance -> undefined, chunk2 -> grants, entity2_end -> 9, entity1_end -> 2, entity2_begin -> 4, entity2 -> OBLIGATION_ACTION, chunk1 -> Fox, sentence -> 0], []]                       |
|[category, 92, 185, GRANTS_TO, [entity1_begin -> 0, relation -> GRANTS_TO, hypothesis -> Fox grants Licensee, confidence -> 0.9822127, nli_prediction -> entail, entity1 -> OBLIGATION_SUBJECT, syntactic_distance -> undefined, chunk2 -> Licensee, entity2_end -> 21, entity1_end -> 2, entity2_begin -> 14, entity2 -> OBLIGATION_INDIRECT_OBJECT, chunk1 -> Fox, sentence -> 0], []]|
+-------------+
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|legre_zero_shot|
|Type:|legal|
|Compatibility:|Legal NLP 1.0.0+|
|License:|Licensed|
|Edition:|Official|
|Language:|en|
|Size:|406.4 MB|
|Case sensitive:|true|

## References

Bert Base (cased) trained on the GLUE MNLI dataset.
