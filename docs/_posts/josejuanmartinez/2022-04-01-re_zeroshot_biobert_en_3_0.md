---
layout: model
title: Zero-shot Relation Extraction (BioBert)
author: John Snow Labs
name: re_zeroshot_biobert
date: 2022-04-01
tags: [zero_shot, zeroshot, zero, shot, en, licensed]
task: Relation Extraction
language: en
edition: Spark NLP for Healthcare 3.4.2
spark_version: 3.0
supported: true
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Zero-shot Relation Extraction to extract relations between clinical entities with no training dataset, just pretrained BioBert embeddings (included in the model).

## Predicted Entities



{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/re_zeroshot_biobert_en_3.4.2_3.0_1648808432473.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
data = spark.createDataFrame(
    [["Paracetamol can alleviate headache or sickness. An MRI test can be used to find cancer."]]
).toDF("text")

documenter = sparknlp.DocumentAssembler() \
    .setInputCol("text") \
    .setOutputCol("document")

tokenizer = sparknlp.annotators.Tokenizer() \
    .setInputCols(["document"]) \
    .setOutputCol("tokens")

sentencer = sparknlp.annotators.SentenceDetector()\
    .setInputCols(["document"])\
    .setOutputCol("sentences")

words_embedder = sparknlp.annotators.WordEmbeddingsModel() \
    .pretrained("embeddings_clinical", "en", "clinical/models") \
    .setInputCols(["sentences", "tokens"]) \
    .setOutputCol("embeddings")

pos_tagger = sparknlp.annotators.PerceptronModel() \
    .pretrained("pos_clinical", "en", "clinical/models") \
    .setInputCols(["sentences", "tokens"]) \
    .setOutputCol("pos_tags")

ner_tagger = sparknlp_jsl.annotator.MedicalNerModel() \
    .pretrained("ner_clinical", "en", "clinical/models") \
    .setInputCols(["sentences", "tokens", "embeddings"]) \
    .setOutputCol("ner_tags")

ner_converter = sparknlp.annotators.NerConverter() \
    .setInputCols(["sentences", "tokens", "ner_tags"]) \
    .setOutputCol("ner_chunks")

dependency_parser = sparknlp.annotators.DependencyParserModel() \
    .pretrained("dependency_conllu", "en") \
    .setInputCols(["document", "pos_tags", "tokens"]) \
    .setOutputCol("dependencies")

re_ner_chunk_filter = sparknlp_jsl.annotator.RENerChunksFilter() \
    .setRelationPairs(["problem-test","problem-treatment"]) \
    .setMaxSyntacticDistance(4)\
    .setDocLevelRelations(False)\
    .setInputCols(["ner_chunks", "dependencies"]) \
    .setOutputCol("re_ner_chunks")

re_model = sparknlp_jsl.annotator.ZeroShotRelationExtractionModel \
    .pretrained("re_zeroshot_biobert", "en", "clinical/models") \
    .setRelationalCategories({
        "CURE": ["{{TREATMENT}} cures {{PROBLEM}}."],
        "IMPROVE": ["{{TREATMENT}} improves {{PROBLEM}}.", "{{TREATMENT}} cures {{PROBLEM}}."],
        "REVEAL": ["{{TEST}} reveals {{PROBLEM}}."]})\
    .setMultiLabel(False)\
    .setInputCols(["re_ner_chunks", "sentences"]) \
    .setOutputCol("relations")

pipeline = sparknlp.base.Pipeline() \
    .setStages([documenter, tokenizer, sentencer, words_embedder, pos_tagger, ner_tagger, ner_converter,
                dependency_parser, re_ner_chunk_filter, re_model])

model = pipeline.fit(data)
results = model.transform(data)

results\
    .selectExpr("explode(relations) as relation")\
    .show(truncate=False)
```

</div>

## Results

```bash
+----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
|relation                                                                                                                                                                                                                                                                                                                                                              |
+----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
|{category, 534, 613, REVEAL, {entity1_begin -> 48, relation -> REVEAL, hypothesis -> An MRI test reveals cancer., confidence -> 0.9760039, nli_prediction -> entail, entity1 -> TEST, syntactic_distance -> 4, chunk2 -> cancer, entity2_end -> 85, entity1_end -> 58, entity2_begin -> 80, entity2 -> PROBLEM, chunk1 -> An MRI test, sentence -> 1}, []}            |
|{category, 267, 357, IMPROVE, {entity1_begin -> 0, relation -> IMPROVE, hypothesis -> Paracetamol improves sickness., confidence -> 0.98819494, nli_prediction -> entail, entity1 -> TREATMENT, syntactic_distance -> 3, chunk2 -> sickness, entity2_end -> 45, entity1_end -> 10, entity2_begin -> 38, entity2 -> PROBLEM, chunk1 -> Paracetamol, sentence -> 0}, []}|
|{category, 0, 90, IMPROVE, {entity1_begin -> 0, relation -> IMPROVE, hypothesis -> Paracetamol improves headache., confidence -> 0.9929625, nli_prediction -> entail, entity1 -> TREATMENT, syntactic_distance -> 2, chunk2 -> headache, entity2_end -> 33, entity1_end -> 10, entity2_begin -> 26, entity2 -> PROBLEM, chunk1 -> Paracetamol, sentence -> 0}, []}    |
+----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|re_zeroshot_biobert|
|Compatibility:|Spark NLP for Healthcare 3.4.2+|
|License:|Licensed|
|Edition:|Official|
|Language:|en|
|Size:|1.3 GB|
|Case sensitive:|true|

## References

As it is a zero-shot relation extractor, no training dataset is necessary.

## Benchmarking

```bash
Since it does not have a training dataset, the accuracy of the model depends on the relations you set in `setRelationalCategories`.
```