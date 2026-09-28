{%- capture title -%}
DolphinForDocumentParsing
{%- endcapture -%}

{%- capture description -%}
Document image parsing with ByteDance Dolphin.

Dolphin follows an analyze-then-parse design. Stage 1 generates the page's layout elements in
reading order; each element is then cropped and parsed independently in stage 2, batched by type.
Tables come back as HTML, formulas as LaTeX, and text as markdown. Because parsing happens
per element rather than as one long generation over the page, output stays on task and the
elements decode in parallel.

Unlike a chat-style vision model, the task prompt is the **decoder prefix**: the image goes to the
encoder and the model continues the prompt. There is no chat template and no image placeholder
token.

Dolphin 1.5 and 1.0 load the same way. The release is recognised from its stage-1 output, which
decides the box geometry and whether formulas and code get their own prompts (1.5 only).

`setParsingMode` selects what you are handing the annotator, not which capability to use — `page`
already parses tables, formulas and text:

| mode | input | work done |
|---|---|---|
| `page` (default) | a whole page | stage 1 + stage 2 |
| `layout` | a whole page | stage 1 only — detect regions without transcribing them |
| `table` / `text` / `formula` / `code` | an already-cropped region | stage 2 only; `formula` and `code` are Dolphin 1.5 prompts |

Generation stops at the decoder's 4096-position window.

Pretrained models can be loaded with `pretrained` of the companion object:

```scala
val dolphin = DolphinForDocumentParsing.pretrained()
  .setInputCols("image_assembler")
  .setOutputCol("elements")
```

The default model is `"dolphin_1_5"`.

With `outputFormat` set to `elements` (the default) each annotation carries `dolphinLabel`,
`elementType`, `readingOrder`, `readingGroup` and `bbox` in its metadata, and tables additionally
carry `tableJson` in the same shape `Reader2Table` produces. Setting it to `markdown` emits a single
assembled document per page instead.

Note that the task comes from `parsingMode`, not from the image annotation's `text` field —
`Reader2Image` defaults its `promptTemplate` to `qwen2vl-chat`, and that markup would otherwise be
fed to Dolphin as a task prompt.
{%- endcapture -%}

{%- capture input_anno -%}
IMAGE
{%- endcapture -%}

{%- capture output_anno -%}
DOCUMENT
{%- endcapture -%}

{%- capture python_example -%}
import sparknlp
from sparknlp.base import *
from sparknlp.annotator import *
from pyspark.ml import Pipeline

imageDF = spark.read.format("image").load(path="./pages/")

imageAssembler = ImageAssembler() \
    .setInputCol("image") \
    .setOutputCol("image_assembler")

dolphin = DolphinForDocumentParsing.pretrained() \
    .setInputCols("image_assembler") \
    .setOutputCol("elements") \
    .setParsingMode("page")

pipeline = Pipeline().setStages([imageAssembler, dolphin])
result = pipeline.fit(imageDF).transform(imageDF)

result.selectExpr("explode(elements) as e") \
    .selectExpr("e.metadata['dolphinLabel'] as label", "e.result") \
    .show(truncate=60)
+-----+------------------------------------------------------------+
|label|                                                      result|
+-----+------------------------------------------------------------+
|sec_0|        LLaMA: Open and Efficient Foundation Language Models|
| para|Hugo Touvron; Thibaut Lavril; Gautier Izacard; Xavier Mar...|
| para|                                                     Meta AI|
|sec_1|                                                    Abstract|
+-----+------------------------------------------------------------+
{%- endcapture -%}

{%- capture scala_example -%}
import com.johnsnowlabs.nlp.ImageAssembler
import com.johnsnowlabs.nlp.annotator.DolphinForDocumentParsing
import org.apache.spark.ml.Pipeline

val imageDF = spark.read.format("image").load("./pages/")

val imageAssembler = new ImageAssembler()
  .setInputCol("image")
  .setOutputCol("image_assembler")

val dolphin = DolphinForDocumentParsing.pretrained()
  .setInputCols("image_assembler")
  .setOutputCol("elements")
  .setParsingMode("page")

val pipeline = new Pipeline().setStages(Array(imageAssembler, dolphin))
val result = pipeline.fit(imageDF).transform(imageDF)

result
  .selectExpr("explode(elements) as e")
  .selectExpr("e.metadata['dolphinLabel'] as label", "e.result")
  .show(truncate = 60)
{%- endcapture -%}

{%- capture api_link -%}
[DolphinForDocumentParsing](/api/com/johnsnowlabs/nlp/annotators/cv/DolphinForDocumentParsing)
{%- endcapture -%}

{%- capture python_api_link -%}
[DolphinForDocumentParsing](/api/python/reference/autosummary/sparknlp/annotator/cv/dolphin_for_document_parsing/index.html#sparknlp.annotator.cv.dolphin_for_document_parsing.DolphinForDocumentParsing)
{%- endcapture -%}

{%- capture source_link -%}
[DolphinForDocumentParsing](https://github.com/JohnSnowLabs/spark-nlp/tree/master/src/main/scala/com/johnsnowlabs/nlp/annotators/cv/DolphinForDocumentParsing.scala)
{%- endcapture -%}

{% include templates/anno_template.md
title=title
description=description
input_anno=input_anno
output_anno=output_anno
python_example=python_example
scala_example=scala_example
api_link=api_link
python_api_link=python_api_link
source_link=source_link
%}
