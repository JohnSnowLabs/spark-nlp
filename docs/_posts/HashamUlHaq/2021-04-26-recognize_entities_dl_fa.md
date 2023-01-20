---
layout: model
title: Explain Document DL Pipeline for Farsi/Persian
author: John Snow Labs
name: recognize_entities_dl
date: 2021-04-26
tags: [pipeline, ner, fa, open_source]
task: Pipeline Public
language: fa
edition: Spark NLP 3.0.0
spark_version: 3.0
supported: true
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

The explain_document_dl is a pretrained pipeline that we can use to process text with a simple pipeline that performs basic processing steps and recognizes entities . It performs most of the common text processing tasks on your dataframe

{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/public/NER_FA/){:.button.button-orange}
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Public/3.SparkNLP_Pretrained_Models.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/recognize_entities_dl_fa_3.0.0_3.0_1619451815476.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/recognize_entities_dl_fa_3.0.0_3.0_1619451815476.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
from sparknlp.pretrained import PretrainedPipeline

pipeline = PretrainedPipeline('recognize_entities_dl', lang = 'fa')

annotations =  pipeline.fullAnnotate("""به گزارش خبرنگار ایرنا ، بر اساس تصمیم این مجمع ، محمد قمی نماینده مردم پاکدشت به عنوان رئیس و علی‌اکبر موسوی خوئینی و شمس‌الدین وهابی نمایندگان مردم تهران به عنوان نواب رئیس انتخاب شدند""")[0]

annotations.keys()


```
```scala
val pipeline = new PretrainedPipeline("recognize_entities_dl", lang = "fa")

val result = pipeline.fullAnnotate("""به گزارش خبرنگار ایرنا ، بر اساس تصمیم این مجمع ، محمد قمی نماینده مردم پاکدشت به عنوان رئیس و علی‌اکبر موسوی خوئینی و شمس‌الدین وهابی نمایندگان مردم تهران به عنوان نواب رئیس انتخاب شدند""")(0)

```

{:.nlu-block}
```python
import nlu

text = ["""به گزارش خبرنگار ایرنا ، بر اساس تصمیم این مجمع ، محمد قمی نماینده مردم پاکدشت به عنوان رئیس و علی‌اکبر موسوی خوئینی و شمس‌الدین وهابی نمایندگان مردم تهران به عنوان نواب رئیس انتخاب شدند"""]

result_df = nlu.load('fa.recognize_entities_dl').predict(text)

result_df
```
</div>

## Results

```bash
|    | document                                                                                                                                                                                  | sentence                                                                                                                                                                                  | token     | clean_tokens   | lemma    | pos   | embeddings   | ner   | entities             |
|---:|:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:----------|:---------------|:---------|:------|:-------------|:------|:---------------------|
|  0 | "به گزارش خبرنگار ایرنا ، بر اساس تصمیم این مجمع ، محمد قمی نماینده مردم پاکدشت به عنوان رئیس و علی‌اکبر موسوی خوئینی و شمس‌الدین وهابی نمایندگان مردم تهران به عنوان نواب رئیس انتخاب شدند | "به گزارش خبرنگار ایرنا ، بر اساس تصمیم این مجمع ، محمد قمی نماینده مردم پاکدشت به عنوان رئیس و علی‌اکبر موسوی خوئینی و شمس‌الدین وهابی نمایندگان مردم تهران به عنوان نواب رئیس انتخاب شدند | "         | "              | "        | PUNCT | "            | O     | خبرنگار ایرنا        |
|  1 |                                                                                                                                                                                           |                                                                                                                                                                                           | به        | گزارش          | به       | ADP   | به           | O     | محمد قمی             |
|  2 |                                                                                                                                                                                           |                                                                                                                                                                                           | گزارش     | خبرنگار        | گزارش    | NOUN  | گزارش        | O     | پاکدشت               |
|  3 |                                                                                                                                                                                           |                                                                                                                                                                                           | خبرنگار   | ایرنا          | خبرنگار  | NOUN  | خبرنگار      | B-ORG | علی‌اکبر موسوی خوئینی |
|  4 |                                                                                                                                                                                           |                                                                                                                                                                                           | ایرنا     | ،              | ایرنا    | PROPN | ایرنا        | I-ORG | شمس‌الدین وهابی       |
|  5 |                                                                                                                                                                                           |                                                                                                                                                                                           | ،         | اساس           | ؛        | PUNCT | ،            | O     | تهران                |
|  6 |                                                                                                                                                                                           |                                                                                                                                                                                           | بر        | تصمیم          | بر       | ADP   | بر           | O     |                      |
|  7 |                                                                                                                                                                                           |                                                                                                                                                                                           | اساس      | این            | اساس     | NOUN  | اساس         | O     |                      |
|  8 |                                                                                                                                                                                           |                                                                                                                                                                                           | تصمیم     | مجمع           | تصمیم    | NOUN  | تصمیم        | O     |                      |
|  9 |                                                                                                                                                                                           |                                                                                                                                                                                           | این       | ،              | این      | DET   | این          | O     |                      |
| 10 |                                                                                                                                                                                           |                                                                                                                                                                                           | مجمع      | محمد           | مجمع     | NOUN  | مجمع         | O     |                      |
| 11 |                                                                                                                                                                                           |                                                                                                                                                                                           | ،         | قمی            | ؛        | PUNCT | ،            | O     |                      |
| 12 |                                                                                                                                                                                           |                                                                                                                                                                                           | محمد      | نماینده        | محمد     | PROPN | محمد         | B-PER |                      |
| 13 |                                                                                                                                                                                           |                                                                                                                                                                                           | قمی       | پاکدشت         | قمی      | PROPN | قمی          | I-PER |                      |
| 14 |                                                                                                                                                                                           |                                                                                                                                                                                           | نماینده   | عنوان          | نماینده  | NOUN  | نماینده      | O     |                      |
| 15 |                                                                                                                                                                                           |                                                                                                                                                                                           | مردم      | رئیس           | مردم     | NOUN  | مردم         | O     |                      |
| 16 |                                                                                                                                                                                           |                                                                                                                                                                                           | پاکدشت    | علی‌اکبر        | پاکدشت   | PROPN | پاکدشت       | B-LOC |                      |
| 17 |                                                                                                                                                                                           |                                                                                                                                                                                           | به        | موسوی          | به       | ADP   | به           | O     |                      |
| 18 |                                                                                                                                                                                           |                                                                                                                                                                                           | عنوان     | خوئینی         | عنوان    | NOUN  | عنوان        | O     |                      |
| 19 |                                                                                                                                                                                           |                                                                                                                                                                                           | رئیس      | شمس‌الدین       | رئیس     | NOUN  | رئیس         | O     |                      |
| 20 |                                                                                                                                                                                           |                                                                                                                                                                                           | و         | وهابی          | او       | CCONJ | و            | O     |                      |
| 21 |                                                                                                                                                                                           |                                                                                                                                                                                           | علی‌اکبر   | نمایندگان      | علی‌اکبر  | PROPN | علی‌اکبر      | B-PER |                      |
| 22 |                                                                                                                                                                                           |                                                                                                                                                                                           | موسوی     | تهران          | موسوی    | PROPN | موسوی        | I-PER |                      |
| 23 |                                                                                                                                                                                           |                                                                                                                                                                                           | خوئینی    | عنوان          | خوئینی   | PROPN | خوئینی       | I-PER |                      |
| 24 |                                                                                                                                                                                           |                                                                                                                                                                                           | و         | نواب           | او       | CCONJ | و            | O     |                      |
| 25 |                                                                                                                                                                                           |                                                                                                                                                                                           | شمس‌الدین  | رئیس           | شمس‌الدین | PROPN | شمس‌الدین     | B-PER |                      |
| 26 |                                                                                                                                                                                           |                                                                                                                                                                                           | وهابی     | انتخاب         | وهابی    | PROPN | وهابی        | I-PER |                      |
| 27 |                                                                                                                                                                                           |                                                                                                                                                                                           | نمایندگان |                | نماینده  | NOUN  | نمایندگان    | O     |                      |
| 28 |                                                                                                                                                                                           |                                                                                                                                                                                           | مردم      |                | مردم     | NOUN  | مردم         | O     |                      |
| 29 |                                                                                                                                                                                           |                                                                                                                                                                                           | تهران     |                | تهران    | PROPN | تهران        | B-LOC |                      |
| 30 |                                                                                                                                                                                           |                                                                                                                                                                                           | به        |                | به       | ADP   | به           | O     |                      |
| 31 |                                                                                                                                                                                           |                                                                                                                                                                                           | عنوان     |                | عنوان    | NOUN  | عنوان        | O     |                      |
| 32 |                                                                                                                                                                                           |                                                                                                                                                                                           | نواب      |                | نواب     | NOUN  | نواب         | O     |                      |
| 33 |                                                                                                                                                                                           |                                                                                                                                                                                           | رئیس      |                | رئیس     | NOUN  | رئیس         | O     |                      |
| 34 |                                                                                                                                                                                           |                                                                                                                                                                                           | انتخاب    |                | انتخاب   | NOUN  | انتخاب       | O     |                      |
| 35 |                                                                                                                                                                                           |                                                                                                                                                                                           | شدند      |                | کرد#کن   | VERB  | شدند         | O     |                      |
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|recognize_entities_dl|
|Type:|pipeline|
|Compatibility:|Spark NLP 3.0.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|fa|

## Included Models

- DocumentAssembler
- SentenceDetector
- TokenizerModel
- StopWordsCleaner
- LemmatizerModel
- PerceptronModel
- WordEmbeddingsModel
- NerDLModel
- NerConverter