{%- capture title -%}
SymmetricDelete Spellchecker
{%- endcapture -%}

{%- capture model_description -%}
Symmetric Delete spelling correction algorithm.

The Symmetric Delete spelling correction algorithm reduces the complexity of edit candidate generation and
dictionary lookup for a given Damerau-Levenshtein distance. It is six orders of magnitude faster
(than the standard approach with deletes + transposes + replaces + inserts) and language independent.

Inspired by [SymSpell](https://github.com/wolfgarbe/SymSpell).

Pretrained models can be loaded with `pretrained` of the companion object:
```
val spell = SymmetricDeleteModel.pretrained()
  .setInputCols("token")
  .setOutputCol("spell")
```
The default model is `"spellcheck_sd"`, if no name is provided.
For available pretrained models please see the [Models Hub](https://nlp.johnsnowlabs.com/models?task=Spell+Check).

See [SymmetricDeleteModelTestSpec](https://github.com/JohnSnowLabs/spark-nlp/blob/master/src/test/scala/com/johnsnowlabs/nlp/annotators/spell/symmetric/SymmetricDeleteModelTestSpec.scala) for further reference.
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

spellChecker = SymmetricDeleteModel.pretrained() \
    .setInputCols(["token"]) \
    .setOutputCol("spell")

pipeline = Pipeline().setStages([
    documentAssembler,
    tokenizer,
    spellChecker
])

data = spark.createDataFrame([["spmetimes i wrrite wordz erong."]]).toDF("text")
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
import com.johnsnowlabs.nlp.annotators.spell.symmetric.SymmetricDeleteModel
import org.apache.spark.ml.Pipeline

val documentAssembler = new DocumentAssembler()
  .setInputCol("text")
  .setOutputCol("document")

val tokenizer = new Tokenizer()
  .setInputCols("document")
  .setOutputCol("token")

val spellChecker = SymmetricDeleteModel.pretrained()
  .setInputCols("token")
  .setOutputCol("spell")

val pipeline = new Pipeline().setStages(Array(
  documentAssembler,
  tokenizer,
  spellChecker
))

val data = Seq("spmetimes i wrrite wordz erong.").toDF("text")
val result = pipeline.fit(data).transform(data)
result.select("spell.result").show(false)
+--------------------------------------+
|result                                |
+--------------------------------------+
|[sometimes, i, write, words, wrong, .]|
+--------------------------------------+

{%- endcapture -%}

{%- capture model_api_link -%}
[SymmetricDeleteModel](https://nlp.johnsnowlabs.com/api/com/johnsnowlabs/nlp/annotators/spell/symmetric/SymmetricDeleteModel)
{%- endcapture -%}

{%- capture model_python_api_link -%}
[SymmetricDeleteModel](/api/python/reference/autosummary/python/sparknlp/annotator/spell_check/symmetric_delete/index.html#sparknlp.annotator.spell_check.symmetric_delete.SymmetricDeleteModel)
{%- endcapture -%}

{%- capture model_source_link -%}
[SymmetricDeleteModel](https://github.com/JohnSnowLabs/spark-nlp/tree/master/src/main/scala/com/johnsnowlabs/nlp/annotators/spell/symmetric/SymmetricDeleteModel.scala)
{%- endcapture -%}

{%- capture approach_description -%}
Trains a Symmetric Delete spelling correction algorithm.
Retrieves tokens and utilizes distance metrics to compute possible derived words.

Inspired by [SymSpell](https://github.com/wolfgarbe/SymSpell).

For instantiated/pretrained models, see SymmetricDeleteModel.

See [SymmetricDeleteModelTestSpec](https://github.com/JohnSnowLabs/spark-nlp/blob/master/src/test/scala/com/johnsnowlabs/nlp/annotators/spell/symmetric/SymmetricDeleteModelTestSpec.scala) for further reference.
{%- endcapture -%}

{%- capture approach_input_anno -%}
TOKEN
{%- endcapture -%}

{%- capture approach_output_anno -%}
TOKEN
{%- endcapture -%}

{%- capture approach_python_example -%}
import sparknlp
from sparknlp.base import *
from sparknlp.annotator import *
from pyspark.ml import Pipeline
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

documentAssembler = DocumentAssembler() \
    .setInputCol("text") \
    .setOutputCol("document")

tokenizer = Tokenizer() \
    .setInputCols(["document"]) \
    .setOutputCol("token")

spellChecker = SymmetricDeleteApproach() \
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
import com.johnsnowlabs.nlp.annotators.spell.symmetric.SymmetricDeleteApproach
import org.apache.spark.ml.Pipeline

val documentAssembler = new DocumentAssembler()
  .setInputCol("text")
  .setOutputCol("document")

val tokenizer = new Tokenizer()
  .setInputCols("document")
  .setOutputCol("token")

val spellChecker = new SymmetricDeleteApproach()
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
[SymmetricDeleteApproach](https://nlp.johnsnowlabs.com/api/com/johnsnowlabs/nlp/annotators/spell/symmetric/SymmetricDeleteApproach)
{%- endcapture -%}

{%- capture approach_python_api_link -%}
[SymmetricDeleteApproach](/api/python/reference/autosummary/python/sparknlp/annotator/spell_check/symmetric_delete/index.html#sparknlp.annotator.spell_check.symmetric_delete.SymmetricDeleteApproach)
{%- endcapture -%}

{%- capture approach_source_link -%}
[SymmetricDeleteApproach](https://github.com/JohnSnowLabs/spark-nlp/tree/master/src/main/scala/com/johnsnowlabs/nlp/annotators/spell/symmetric/SymmetricDeleteApproach.scala)
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
