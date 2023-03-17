---
layout: model
title: Clinical Findings to UMLS Code Pipeline
author: John Snow Labs
name: umls_clinical_findings_resolver_pipeline
date: 2023-03-10
tags: [en, licensed, umls, pipeline]
task: Entity Resolution
language: en
edition: Healthcare NLP 4.3.0
spark_version: 3.2
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
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/umls_clinical_findings_resolver_pipeline_en_4.3.0_3.2_1678436541287.zip){:.button.button-orange}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/clinical/models/umls_clinical_findings_resolver_pipeline_en_4.3.0_3.2_1678436541287.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
from sparknlp.pretrained import PretrainedPipeline

pipeline = PretrainedPipeline("umls_clinical_findings_resolver_pipeline", "en", "clinical/models")

text = '''['HTG-induced pancreatitis associated with an acute hepatitis, and obesity']'''

result = pipeline.annotate(text)
```
```scala
import com.johnsnowlabs.nlp.pretrained.PretrainedPipeline

val pipeline = new PretrainedPipeline("umls_clinical_findings_resolver_pipeline", "en", "clinical/models")

val text = "HTG-induced pancreatitis associated with an acute hepatitis, and obesity"

val result = pipeline.annotate(text)
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
|Compatibility:|Healthcare NLP 4.3.0+|
|License:|Licensed|
|Edition:|Official|
|Language:|en|
|Size:|4.3 GB|

## Included Models

- DocumentAssembler
- SentenceDetectorDLModel
- TokenizerModel
- WordEmbeddingsModel
- MedicalNerModel
- NerConverterInternalModel
- ChunkMapperModel
- ChunkMapperModel
- ChunkMapperFilterer
- Chunk2Doc
- BertSentenceEmbeddings
- SentenceEntityResolverModel
- ResolverMerger