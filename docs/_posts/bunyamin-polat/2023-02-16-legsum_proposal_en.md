---
layout: model
title: Legal Proposal Summarization
author: John Snow Labs
name: legsum_proposal
date: 2023-02-16
tags: [en, licensed, summarization, legal, tensorflow]
task: Summarization
language: en
edition: Legal NLP 1.0.0
spark_version: 3.0
supported: true
engine: tensorflow
annotator: T5Transformer
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This model is fine-tuned with a legal dataset (about EU proposals). Summarizes a proposal given on a socially important issue.

## Predicted Entities



{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/legal/models/legsum_proposal_en_1.0.0_3.0_1676587991098.zip){:.button.button-orange}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/legal/models/legsum_proposal_en_1.0.0_3.0_1676587991098.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}

```python
document_assembler = nlp.DocumentAssembler()\
    .setInputCol("text")\
    .setOutputCol("document")

t5 = nlp.T5Transformer().pretrained("legsum_proposal", "en", "legal/models")\
    .setTask("summarize")\
    .setMaxOutputLength(512)\
    .setInputCols(["document"])\
    .setOutputCol("summaries")

text = """The main reason for migration is poverty, and often times it is down to corruption in the leadership of poor countries. What people in such countries demand time and again is that the EU does not engage with their government, and does not supply financial support (which tends to end up in the wrong hands). The EU needs a strict line of engagement. One could envision a rating list by the EU that defines clear requirements support receiving nations must fulfill. Support should be granted in the form of improved economic conditions, such as increased import quota, discounted machinery, and technical know-how injection, not in terms of financial support. Countries failing to fulfill the requirements, especially those with indications of corruption must be put under strict embargoes."""

data_df = spark.createDataFrame([[text]]).toDF("text")

pipeline = nlp.Pipeline().setStages([document_assembler, t5])

results = pipeline.fit(data_df).transform(data_df)

results.select("summaries.result").show(truncate=False)
```

</div>

## Results

```bash
People in poor countries demand that the EU does not engage with their government and do not provide financial support.
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|legsum_proposal|
|Compatibility:|Legal NLP 1.0.0+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[documents]|
|Output Labels:|[summaries]|
|Language:|en|
|Size:|925.9 MB|

## References

Training dataset available [here](https://touche.webis.de/clef23/touche23-web/multilingual-stance-classification.html#data)
