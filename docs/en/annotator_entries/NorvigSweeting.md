{%- capture title -%}
NorvigSweeting Spellchecker
{%- endcapture -%}

{%- capture model_description -%}
This annotator retrieves tokens and makes corrections automatically if not found in an English dictionary.
Inspired by Norvig model and [SymSpell](https://github.com/wolfgarbe/SymSpell).

The Symmetric Delete spelling correction algorithm reduces the complexity of edit candidate generation and
dictionary lookup for a given Damerau-Levenshtein distance. It is six orders of magnitude faster
(than the standard approach with deletes + transposes + replaces + inserts) and language independent.

This is the instantiated model of the NorvigSweetingApproach.
For training your own model, please see the documentation of that class.

Pretrained models can be loaded with `pretrained` of the companion object:
```
val spellChecker = NorvigSweetingModel.pretrained()
  .setInputCols("token")
  .setOutputCol("spell")
  .setDoubleVariants(true)
```
The default model is `"spellcheck_norvig"`, if no name is provided.
For available pretrained models please see the [Models Hub](https://nlp.johnsnowlabs.com/models?task=Spell+Check).


For extended examples of usage, see the [Spark NLP Workshop](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/jupyter/training/english/vivekn-sentiment/VivekNarayanSentimentApproach.ipynb)
and the [NorvigSweetingTestSpec](https://github.com/JohnSnowLabs/spark-nlp/blob/master/src/test/scala/com/johnsnowlabs/nlp/annotators/spell/norvig/NorvigSweetingTestSpec.scala).
{%- endcapture -%}

{%- capture model_input_anno -%}
TOKEN
{%- endcapture -%}

{%- capture model_output_anno -%}
TOKEN
{%- endcapture -%}

{%- capture model_python_example -%}
import sparknlp
from sparknlp.base import *
from sparknlp.annotator import *
from pyspark.ml import Pipeline


documentAssembler = DocumentAssembler() \
    .setInputCol("text") \
    .setOutputCol("document")

tokenizer = Tokenizer() \
    .setInputCols(["document"]) \
    .setOutputCol("token")

spellChecker = NorvigSweetingModel.pretrained() \
    .setInputCols(["token"]) \
    .setOutputCol("spell")

pipeline = Pipeline().setStages([
    documentAssembler,
    tokenizer,
    spellChecker
])

data = spark.createDataFrame([["somtimes i wrrite wordz erong."]]).toDF("text")
result = pipeline.fit(data).transform(data)
result.select("spell.result").show(truncate=False)
+--------------------------------------+
|result                                |
+--------------------------------------+
|[sometimes, i, write, words, wrong, .]|
+--------------------------------------+

{%- endcapture -%}

{%- capture model_scala_example -%}
import spark.implicits._
import com.johnsnowlabs.nlp.base.DocumentAssembler
import com.johnsnowlabs.nlp.annotators.Tokenizer
import com.johnsnowlabs.nlp.annotators.spell.norvig.NorvigSweetingModel

import org.apache.spark.ml.Pipeline

val documentAssembler = new DocumentAssembler()
  .setInputCol("text")
  .setOutputCol("document")

val tokenizer = new Tokenizer()
  .setInputCols("document")
  .setOutputCol("token")

val spellChecker = NorvigSweetingModel.pretrained()
  .setInputCols("token")
  .setOutputCol("spell")

val pipeline = new Pipeline().setStages(Array(
  documentAssembler,
  tokenizer,
  spellChecker
))

val data = Seq("somtimes i wrrite wordz erong.").toDF("text")
val result = pipeline.fit(data).transform(data)
result.select("spell.result").show(false)
+--------------------------------------+
|result                                |
+--------------------------------------+
|[sometimes, i, write, words, wrong, .]|
+--------------------------------------+

{%- endcapture -%}

{%- capture model_api_link -%}
[NorvigSweetingModel](https://nlp.johnsnowlabs.com/api/com/johnsnowlabs/nlp/annotators/spell/norvig/NorvigSweetingModel)
{%- endcapture -%}

{%- capture model_python_api_link -%}
[NorvigSweetingModel](/api/python/reference/autosummary/python/sparknlp/annotator/spell_check/norvig_sweeting/index.html#sparknlp.annotator.spell_check.norvig_sweeting.NorvigSweetingModel)
{%- endcapture -%}

{%- capture model_source_link -%}
[NorvigSweetingModel](https://github.com/JohnSnowLabs/spark-nlp/tree/master/src/main/scala/com/johnsnowlabs/nlp/annotators/spell/norvig/NorvigSweetingModel.scala)
{%- endcapture -%}

{%- capture approach_description -%}
Trains annotator, that retrieves tokens and makes corrections automatically if not found in an English dictionary.

The Symmetric Delete spelling correction algorithm reduces the complexity of edit candidate generation and
dictionary lookup for a given Damerau-Levenshtein distance. It is six orders of magnitude faster
(than the standard approach with deletes + transposes + replaces + inserts) and language independent.
A dictionary of correct spellings must be provided with `setDictionary` as a text file, where each word is parsed by a regex pattern.

Inspired by Norvig model and [SymSpell](https://github.com/wolfgarbe/SymSpell).

For instantiated/pretrained models, see NorvigSweetingModel.

For extended examples of usage, see the [Spark NLP Workshop](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/jupyter/training/english/vivekn-sentiment/VivekNarayanSentimentApproach.ipynb)
and the [NorvigSweetingTestSpec](https://github.com/JohnSnowLabs/spark-nlp/blob/master/src/test/scala/com/johnsnowlabs/nlp/annotators/spell/norvig/NorvigSweetingTestSpec.scala).
{%- endcapture -%}

{%- capture approach_input_anno -%}
TOKEN
{%- endcapture -%}

{%- capture approach_output_anno -%}
TOKEN
{%- endcapture -%}

{%- capture approach_python_example -%}
# In this example, the dictionary `"words.txt"` has the form of
#
# ...
# gummy
# gummic
# gummier
# gummiest
# gummiferous
# ...
#
# This dictionary is then set to be the basis of the spell checker.

import sparknlp
from sparknlp.base import *
from sparknlp.annotator import *
from pyspark.ml import Pipeline

documentAssembler = DocumentAssembler() \
    .setInputCol("text") \
    .setOutputCol("document")

tokenizer = Tokenizer() \
    .setInputCols(["document"]) \
    .setOutputCol("token")

spellChecker = NorvigSweetingApproach() \
    .setInputCols(["token"]) \
    .setOutputCol("spell") \
    .setDictionary("src/test/resources/spell/words.txt")

pipeline = Pipeline().setStages([
    documentAssembler,
    tokenizer,
    spellChecker
])

pipelineModel = pipeline.fit(trainingData)

{%- endcapture -%}

{%- capture approach_scala_example -%}
// In this example, the dictionary `"words.txt"` has the form of
//
// ...
// gummy
// gummic
// gummier
// gummiest
// gummiferous
// ...
//
// This dictionary is then set to be the basis of the spell checker.

import com.johnsnowlabs.nlp.base.DocumentAssembler
import com.johnsnowlabs.nlp.annotators.Tokenizer
import com.johnsnowlabs.nlp.annotators.spell.norvig.NorvigSweetingApproach
import org.apache.spark.ml.Pipeline

val documentAssembler = new DocumentAssembler()
  .setInputCol("text")
  .setOutputCol("document")

val tokenizer = new Tokenizer()
  .setInputCols("document")
  .setOutputCol("token")

val spellChecker = new NorvigSweetingApproach()
  .setInputCols("token")
  .setOutputCol("spell")
  .setDictionary("src/test/resources/spell/words.txt")

val pipeline = new Pipeline().setStages(Array(
  documentAssembler,
  tokenizer,
  spellChecker
))

val pipelineModel = pipeline.fit(trainingData)

{%- endcapture -%}

{%- capture approach_api_link -%}
[NorvigSweetingApproach](https://nlp.johnsnowlabs.com/api/com/johnsnowlabs/nlp/annotators/spell/norvig/NorvigSweetingApproach)
{%- endcapture -%}

{%- capture approach_python_api_link -%}
[NorvigSweetingApproach](/api/python/reference/autosummary/python/sparknlp/annotator/spell_check/norvig_sweeting/index.html#sparknlp.annotator.spell_check.norvig_sweeting.NorvigSweetingApproach)
{%- endcapture -%}

{%- capture approach_source_link -%}
[NorvigSweetingApproach](https://github.com/JohnSnowLabs/spark-nlp/tree/master/src/main/scala/com/johnsnowlabs/nlp/annotators/spell/norvig/NorvigSweetingApproach.scala)
{%- endcapture -%}


{% include templates/approach_model_template.md
title=title
model_description=model_description
model_input_anno=model_input_anno
model_output_anno=model_output_anno
model_python_api_link=model_python_api_link
model_api_link=model_api_link
model_source_link=model_source_link
approach_description=approach_description
approach_input_anno=approach_input_anno
approach_output_anno=approach_output_anno
approach_python_example=approach_python_example
approach_scala_example=approach_scala_example
approach_python_api_link=approach_python_api_link
approach_api_link=approach_api_link
approach_source_link=approach_source_link
%}
