{%- capture title -%}
LanguageDetectorDL
{%- endcapture -%}

{%- capture description -%}
Language Identification and Detection by using CNN and RNN architectures in TensorFlow, or supervised
fastText models such as GlotLID on ONNX.

`LanguageDetectorDL` is an annotator that detects the language of documents or sentences depending on the inputCols.
The CNN models are trained on large datasets such as Wikipedia and Tatoeba.
Depending on the language (how similar the characters are), they work
best with text longer than 140 characters.
Their output is a language code in [Wiki Code style](https://en.wikipedia.org/wiki/List_of_Wikipedias).

The fastText models run on ONNX and turn text into features exactly as fastText does. GlotLID v4
(`"ld_glotlid_v4"`, or `"ld_glotlid_v4_int8"` at a quarter of the size) labels 2,060 languages and scripts as an
ISO 639-3 code plus an ISO 15924 script, such as `eng_Latn` or `cmn_Hani`, and has `zxx_Code`, `zxx_Zmth` and
`zxx_Zsym` for code, mathematics and symbols. For these models the metadata also holds `confidence`, the score the
label was judged on, and three parameters shape the result:

- `scriptPooling`: for a non-Latin top label, the score is the sum of all labels in its script, so that close
  varieties such as Mandarin (`cmn_Hani`) and Cantonese (`yue_Hani`) do not split it; the label is still the top
  one, so the pooled score says how sure the script is rather than the language
- `minLetters`: when the top label is in Latin script (`_Latn`), text with fewer letters than this gets
  `thresholdLabel`
- `topK`: number of scores kept in the metadata, the highest ones

The pretrained GlotLID models are saved with `threshold` 0.7, `thresholdLabel` `"unk"`, `scriptPooling`,
`minLetters` 20, `topK` 5 and one result per sentence (`coalesceSentences` false). Each sentence goes to the model
whole, so split long documents into sentences first. To import other fastText models,
see the [import notebook](https://github.com/JohnSnowLabs/spark-nlp/blob/master/examples/python/transformers/onnx/HuggingFace_ONNX_in_Spark_NLP_LanguageDetectorDL.ipynb);
for finding the non-English sentences of a collection, see the
[GlotLID example](https://github.com/JohnSnowLabs/spark-nlp/blob/master/examples/python/annotation/text/english/language-detection/Language_Detection_with_GlotLID.ipynb).

Pretrained models can be loaded with `pretrained` of the companion object:
```
Val languageDetector = LanguageDetectorDL.pretrained()
  .setInputCols("sentence")
  .setOutputCol("language")
```
The default model is `"ld_wiki_tatoeba_cnn_21"`, default language is `"xx"` (meaning multi-lingual),
if no values are provided.
For available pretrained models please see the [Models Hub](https://sparknlp.org/models?task=Language+Detection).

For extended examples of usage, see the [Examples](https://github.com/JohnSnowLabs/spark-nlp/blob/master/examples/python/annotation/text/english/language-detection/Language_Detection_and_Indentification.ipynb)
And the [LanguageDetectorDLTestSpec](https://github.com/JohnSnowLabs/spark-nlp/blob/master/src/test/scala/com/johnsnowlabs/nlp/annotators/ld/dl/LanguageDetectorDLTestSpec.scala).
{%- endcapture -%}

{%- capture input_anno -%}
DOCUMENT
{%- endcapture -%}

{%- capture output_anno -%}
LANGUAGE
{%- endcapture -%}

{%- capture python_example -%}
import sparknlp
from sparknlp.base import *
from sparknlp.annotator import *
from pyspark.ml import Pipeline

documentAssembler = DocumentAssembler() \
    .setInputCol("text") \
    .setOutputCol("document")

languageDetector = LanguageDetectorDL.pretrained() \
    .setInputCols("document") \
    .setOutputCol("language")

pipeline = Pipeline() \
    .setStages([
      documentAssembler,
      languageDetector
    ])

data = spark.createDataFrame([
    ["Spark NLP is an open-source text processing library for advanced natural language processing for the Python, Java and Scala programming languages."],
    ["Spark NLP est une bibliothèque de traitement de texte open source pour le traitement avancé du langage naturel pour les langages de programmation Python, Java et Scala."],
    ["Spark NLP ist eine Open-Source-Textverarbeitungsbibliothek für fortgeschrittene natürliche Sprachverarbeitung für die Programmiersprachen Python, Java und Scala."]
]).toDF("text")
result = pipeline.fit(data).transform(data)

result.select("language.result").show(truncate=False)
+------+
|result|
+------+
|[en]  |
|[fr]  |
|[de]  |
+------+

{%- endcapture -%}

{%- capture scala_example -%}
import spark.implicits._
import com.johnsnowlabs.nlp.base.DocumentAssembler
import com.johnsnowlabs.nlp.annotators.ld.dl.LanguageDetectorDL
import org.apache.spark.ml.Pipeline

val documentAssembler = new DocumentAssembler()
  .setInputCol("text")
  .setOutputCol("document")

val languageDetector = LanguageDetectorDL.pretrained()
  .setInputCols("document")
  .setOutputCol("language")

val pipeline = new Pipeline()
  .setStages(Array(
    documentAssembler,
    languageDetector
  ))

val data = Seq(
  "Spark NLP is an open-source text processing library for advanced natural language processing for the Python, Java and Scala programming languages.",
  "Spark NLP est une bibliothèque de traitement de texte open source pour le traitement avancé du langage naturel pour les langages de programmation Python, Java et Scala.",
  "Spark NLP ist eine Open-Source-Textverarbeitungsbibliothek für fortgeschrittene natürliche Sprachverarbeitung für die Programmiersprachen Python, Java und Scala."
).toDF("text")
val result = pipeline.fit(data).transform(data)

result.select("language.result").show(false)
+------+
|result|
+------+
|[en]  |
|[fr]  |
|[de]  |
+------+

{%- endcapture -%}

{%- capture api_link -%}
[LanguageDetectorDL](/api/com/johnsnowlabs/nlp/annotators/ld/dl/LanguageDetectorDL)
{%- endcapture -%}

{%- capture python_api_link -%}
[LanguageDetectorDL](/api/python/reference/autosummary/sparknlp/annotator/ld_dl/language_detector_dl/index.html#sparknlp.annotator.ld_dl.language_detector_dl.LanguageDetectorDL)
{%- endcapture -%}

{%- capture source_link -%}
[LanguageDetectorDL](https://github.com/JohnSnowLabs/spark-nlp/tree/master/src/main/scala/com/johnsnowlabs/nlp/annotators/ld/dl/LanguageDetectorDL.scala)
{%- endcapture -%}

{% include templates/anno_template.md
title=title
description=description
input_anno=input_anno
output_anno=output_anno
python_example=python_example
scala_example=scala_example
python_api_link=python_api_link
api_link=api_link
source_link=source_link
%}