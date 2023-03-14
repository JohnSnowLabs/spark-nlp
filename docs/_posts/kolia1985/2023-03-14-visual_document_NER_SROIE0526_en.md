---
layout: model
title: Visual Document NER fine-tuned on SROIE
author: John Snow Labs
name: visual_document_NER_SROIE0526
date: 2023-03-14
tags: [en, licensed]
task: Named Entity Recognition
language: en
edition: Visual NLP 4.3.1
spark_version: [3.0, 3.2]
supported: true
annotator: VisualDocumentNerLayoutLMv1
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This visual NER model is based on LayoutLM pre-trained model and fine-tuned with SROIE dataset

## Predicted Entities



{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/ocr/visual_document_NER_SROIE0526_en_4.3.1_3.2_1678787416820.zip){:.button.button-orange}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/clinical/ocr/visual_document_NER_SROIE0526_en_4.3.1_3.2_1678787416820.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use

ocr = ImageToHocr()\
            .setInputCol("image")\
            .setOutputCol("hocr")\
            .setIgnoreResolution(False)\
            .setOcrParams(["preserve_interword_spaces=0"])

        doc_ner = VisualDocumentNer()\
            .pretrained("visual_document_NER_SROIE0526", "en", "clinical/ocr") \
            .setInputCol("hocr")\
            .setOutputCol("label")

        df = doc_ner.transform(ocr.transform(visual_document_df))
        path_array = split(df['path'], '/')
        df.withColumn('filename', path_array.getItem(size(path_array) - 1)) \
            .select("filename", "entities", "label") \
            .show(truncate=False)

<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
       ocr = ImageToHocr()\
            .setInputCol("image")\
            .setOutputCol("hocr")\
            .setIgnoreResolution(False)\
            .setOcrParams(["preserve_interword_spaces=0"])

        doc_ner = VisualDocumentNer()\
            .pretrained("visual_document_NER_SROIE0526", "en", "clinical/ocr") \
            .setInputCol("hocr")\
            .setOutputCol("label")

        df = doc_ner.transform(ocr.transform(visual_document_df))
        path_array = split(df['path'], '/')
        df.withColumn('filename', path_array.getItem(size(path_array) - 1)) \
            .select("filename", "entities", "label") \
            .show(truncate=False)
```

</div>

## Results

```bash
+------------------------------------------------------------------------+---------+
|entities                                                                |label    |
+------------------------------------------------------------------------+---------+
|[entity, 0, 0, O, [word -> [1060, token -> [], []]                      |O        |
|[entity, 0, 0, O, [word -> [1060, token -> 1060], []]                   |O        |
|[entity, 0, 0, O, [word -> [1060, token -> 1060], []]                   |O        |
|[entity, 0, 0, O, [word -> 257, token -> 257], []]                      |O        |
|[entity, 0, 0, O, [word -> LEMON, token -> lemon], []]                  |O        |
|[entity, 0, 0, O, [word -> TREE, token -> tree], []]                    |O        |
|[entity, 0, 0, B-COMPANY, [word -> RESTAURANT, token -> restaurant], []]|B-COMPANY|
|[entity, 0, 0, B-COMPANY, [word -> JTJ, token -> jtj], []]              |B-COMPANY|
|[entity, 0, 0, B-COMPANY, [word -> JTJ, token -> jtj], []]              |B-COMPANY|
|[entity, 0, 0, B-COMPANY, [word -> JTJ, token -> jtj], []]              |B-COMPANY|
|[entity, 0, 0, B-COMPANY, [word -> FOODS, token -> foods], []]          |B-COMPANY|
|[entity, 0, 0, B-COMPANY, [word -> SDN, token -> sdn], []]              |B-COMPANY|
|[entity, 0, 0, B-COMPANY, [word -> SDN, token -> sdn], []]              |B-COMPANY|
|[entity, 0, 0, B-COMPANY, [word -> BHD, token -> bhd], []]              |B-COMPANY|
|[entity, 0, 0, B-COMPANY, [word -> BHD, token -> bhd], []]              |B-COMPANY|
|[entity, 0, 0, O, [word -> (1179227A), token -> (], []]                 |O        |
|[entity, 0, 0, O, [word -> (1179227A), token -> 1179227a], []]          |O        |
|[entity, 0, 0, O, [word -> (1179227A), token -> 1179227a], []]          |O        |
|[entity, 0, 0, O, [word -> (1179227A), token -> 1179227a], []]          |O        |
|[entity, 0, 0, O, [word -> (1179227A), token -> 1179227a], []]          |O        |
|[entity, 0, 0, O, [word -> (1179227A), token -> 1179227a], []]          |O        |
|[entity, 0, 0, O, [word -> (1179227A), token -> )], []]                 |O        |
|[entity, 0, 0, O, [word -> GST, token -> gst], []]                      |O        |
|[entity, 0, 0, O, [word -> GST, token -> gst], []]                      |O        |
|[entity, 0, 0, O, [word -> 001085747200, token -> 001085747200], []]    |O        |
|[entity, 0, 0, O, [word -> 001085747200, token -> 001085747200], []]    |O        |
|[entity, 0, 0, O, [word -> 001085747200, token -> 001085747200], []]    |O        |
|[entity, 0, 0, O, [word -> 001085747200, token -> 001085747200], []]    |O        |
|[entity, 0, 0, O, [word -> 001085747200, token -> 001085747200], []]    |O        |
|[entity, 0, 0, O, [word -> 001085747200, token -> 001085747200], []]    |O        |
|[entity, 0, 0, O, [word -> No, token -> no], []]                        |O        |
|[entity, 0, 0, O, [word -> 3,, token -> 3], []]                         |O        |
|[entity, 0, 0, O, [word -> 3,, token -> ,], []]                         |O        |
|[entity, 0, 0, O, [word -> Jalan, token -> jalan], []]                  |O        |
|[entity, 0, 0, O, [word -> Permas, token -> permas], []]                |O        |
|[entity, 0, 0, O, [word -> Permas, token -> permas], []]                |O        |
|[entity, 0, 0, O, [word -> 10/8,, token -> 10], []]                     |O        |
|[entity, 0, 0, O, [word -> 10/8,, token -> /], []]                      |O        |
|[entity, 0, 0, O, [word -> 10/8,, token -> 8], []]                      |O        |
|[entity, 0, 0, O, [word -> 10/8,, token -> ,], []]                      |O        |
|[entity, 0, 0, O, [word -> Bandar, token -> bandar], []]                |O        |
|[entity, 0, 0, O, [word -> Bandar, token -> bandar], []]                |O        |
|[entity, 0, 0, O, [word -> Baru, token -> baru], []]                    |O        |
|[entity, 0, 0, O, [word -> Baru, token -> baru], []]                    |O        |
|[entity, 0, 0, O, [word -> Perrnas, token -> perrnas], []]              |O        |
|[entity, 0, 0, O, [word -> Perrnas, token -> perrnas], []]              |O        |
|[entity, 0, 0, O, [word -> Perrnas, token -> perrnas], []]              |O        |
|[entity, 0, 0, O, [word -> Jaya,, token -> jaya], []]                   |O        |
|[entity, 0, 0, O, [word -> Jaya,, token -> ,], []]                      |O        |
|[entity, 0, 0, O, [word -> 81750, token -> 81750], []]                  |O        |
|[entity, 0, 0, O, [word -> 81750, token -> 81750], []]                  |O        |
|[entity, 0, 0, O, [word -> 81750, token -> 81750], []]                  |O        |
|[entity, 0, 0, O, [word -> Masai,, token -> masai], []]                 |O        |
|[entity, 0, 0, O, [word -> Masai,, token -> masai], []]                 |O        |
|[entity, 0, 0, O, [word -> Masai,, token -> ,], []]                     |O        |
|[entity, 0, 0, O, [word -> Johor, token -> johor], []]                  |O        |
|[entity, 0, 0, O, [word -> 07, token -> 07], []]                        |O        |
|[entity, 0, 0, O, [word -> 3823456, token -> 3823456], []]              |O        |
|[entity, 0, 0, O, [word -> 3823456, token -> 3823456], []]              |O        |
|[entity, 0, 0, O, [word -> 3823456, token -> 3823456], []]              |O        |
|[entity, 0, 0, O, [word -> 3823456, token -> 3823456], []]              |O        |
|[entity, 0, 0, O, [word -> SIMPLIFIED, token -> simplified], []]        |O        |
|[entity, 0, 0, O, [word -> TAX, token -> tax], []]                      |O        |
|[entity, 0, 0, O, [word -> INVOICE, token -> invoice], []]              |O        |
|[entity, 0, 0, O, [word -> INVOICE, token -> invoice], []]              |O        |
|[entity, 0, 0, O, [word -> INVOICE, token -> invoice], []]              |O        |
|[entity, 0, 0, O, [word -> INVOICENO, token -> invoiceno], []]          |O        |
|[entity, 0, 0, O, [word -> INVOICENO, token -> invoiceno], []]          |O        |
|[entity, 0, 0, O, [word -> INVOICENO, token -> invoiceno], []]          |O        |
|[entity, 0, 0, O, [word -> INVOICENO, token -> invoiceno], []]          |O        |
|[entity, 0, 0, O, [word ->  ©S00014/69, token -> ©s00014], []]          |O        |
|[entity, 0, 0, O, [word ->  ©S00014/69, token -> ©s00014], []]          |O        |
|[entity, 0, 0, O, [word ->  ©S00014/69, token -> ©s00014], []]          |O        |
|[entity, 0, 0, O, [word ->  ©S00014/69, token -> ©s00014], []]          |O        |
|[entity, 0, 0, O, [word ->  ©S00014/69, token -> ©s00014], []]          |O        |
|[entity, 0, 0, O, [word ->  ©S00014/69, token -> /], []]                |O        |
|[entity, 0, 0, O, [word ->  ©S00014/69, token -> 69], []]               |O        |
|[entity, 0, 0, O, [word -> INVOICE, token -> invoice], []]              |O        |
|[entity, 0, 0, O, [word -> INVOICE, token -> invoice], []]              |O        |
|[entity, 0, 0, O, [word -> INVOICE, token -> invoice], []]              |O        |
|[entity, 0, 0, O, [word -> DALE:, token -> dale], []]                   |O        |
|[entity, 0, 0, O, [word -> DALE:, token -> :], []]                      |O        |
|[entity, 0, 0, B-DATE, [word -> 6/1/2018, token -> 6], []]              |B-DATE   |
|[entity, 0, 0, O, [word -> 6/1/2018, token -> /], []]                   |O        |
|[entity, 0, 0, B-DATE, [word -> 6/1/2018, token -> 1], []]              |B-DATE   |
|[entity, 0, 0, O, [word -> 6/1/2018, token -> /], []]                   |O        |
|[entity, 0, 0, B-DATE, [word -> 6/1/2018, token -> 2018], []]           |B-DATE   |
|[entity, 0, 0, O, [word -> 6:42:02, token -> 6], []]                    |O        |
|[entity, 0, 0, O, [word -> 6:42:02, token -> :], []]                    |O        |
|[entity, 0, 0, O, [word -> 6:42:02, token -> 42], []]                   |O        |
|[entity, 0, 0, O, [word -> 6:42:02, token -> :], []]                    |O        |
|[entity, 0, 0, O, [word -> 6:42:02, token -> 02], []]                   |O        |
|[entity, 0, 0, O, [word -> PM, token -> pm], []]                        |O        |
|[entity, 0, 0, O, [word -> WAITER:, token -> waiter], []]               |O        |
|[entity, 0, 0, O, [word -> WAITER:, token -> :], []]                    |O        |
|[entity, 0, 0, O, [word -> Vanessa, token -> vanessa], []]              |O        |
|[entity, 0, 0, O, [word -> “Sane, token -> “sane], []]                  |O        |
|[entity, 0, 0, O, [word -> “Sane, token -> “sane], []]                  |O        |
|[entity, 0, 0, O, [word -> “Sane, token -> “sane], []]                  |O        |
|[entity, 0, 0, O, [word -> Pax, token -> pax], []]                      |O        |
+------------------------------------------------------------------------+---------+
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|visual_document_NER_SROIE0526|
|Type:|ocr|
|Compatibility:|Visual NLP 4.3.1+|
|License:|Licensed|
|Edition:|Official|
|Language:|en|
|Size:|418.4 MB|
|Case sensitive:|false|
|Max sentence length:|512|