---
layout: model
title: Clinical Findings to UMLS Code Pipeline
author: John Snow Labs
name: umls_clinical_findings_resolver_pipeline
date: 2022-07-26
tags: [en, licensed, umls, pipeline]
task: Pipeline Healthcare
language: en
edition: Healthcare NLP 4.0.0
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This pretrained pipeline maps entities (Clinical Findings) with their corresponding UMLS CUI codes. You’ll just feed your text and it will return the corresponding UMLS codes.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/26.Chunk_Mapping.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/umls_clinical_findings_resolver_pipeline_en_4.0.0_3.0_1658822255140.zip){:.button.button-orange.button-orange-trans.arr.button-icon.hidden}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/clinical/models/umls_clinical_findings_resolver_pipeline_en_4.0.0_3.0_1658822255140.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
from sparknlp.pretrained import PretrainedPipeline

pipeline= PretrainedPipeline("umls_clinical_findings_resolver_pipeline", "en", "clinical/models")
pipeline.annotate("HTG-induced pancreatitis associated with an acute hepatitis, and obesity")
```
```scala
import com.johnsnowlabs.nlp.pretrained.PretrainedPipeline

val pipeline= PretrainedPipeline("umls_clinical_findings_resolver_pipeline", "en", "clinical/models")
val pipeline.annotate("HTG-induced pancreatitis associated with an acute hepatitis, and obesity")
```
</div>

## Results

```bash
+------------------------+---------+---------+
|chunk                   |ner_label|umls_code|
+------------------------+---------+---------+
|HTG-induced pancreatitis|PROBLEM  |C1963198 |
|an acute hepatitis      |PROBLEM  |C4750596 |
|obesity                 |PROBLEM  |C1963185 |
+------------------------+---------+---------+
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|umls_clinical_findings_resolver_pipeline|
|Type:|pipeline|
|Compatibility:|Healthcare NLP 4.0.0+|
|License:|Licensed|
|Edition:|Official|
|Language:|en|
|Size:|4.3 GB|

## Included Models

- DocumentAssembler
- SentenceDetector
- TokenizerModel
- WordEmbeddingsModel
- MedicalNerModel
- NerConverter
- ChunkMapperModel
- ChunkMapperModel
- ChunkMapperFilterer
- Chunk2Doc
- BertSentenceEmbeddings
- SentenceEntityResolverModel
- ResolverMerger