{%- capture title -%}
SymmetricDelete Spellchecker
{%- endcapture -%}

{%- capture description -%}
Trains a Symmetric Delete spelling correction algorithm.
Retrieves tokens and utilizes distance metrics to compute possible derived words.

A dictionary of correct spellings must be provided with `setDictionary` as a text file, where each word is parsed by a regex pattern.

For Example a file `"words.txt"`:
```
...
gummy
gummic
gummier
gummiest
gummiferous
...
```
can be parsed with the regular expression `\S+`, which is the default for `setDictionary`.

This dictionary is then set to be the basis of the spell checker.

Inspired by [SymSpell](https://github.com/wolfgarbe/SymSpell).

For instantiated/pretrained models, see SymmetricDeleteModel.

See [SymmetricDeleteModelTestSpec](https://github.com/JohnSnowLabs/spark-nlp/blob/master/src/test/scala/com/johnsnowlabs/nlp/annotators/spell/symmetric/SymmetricDeleteModelTestSpec.scala) for further reference.
{%- endcapture -%}

{%- capture input_anno -%}
TOKEN
{%- endcapture -%}

{%- capture output_anno -%}
TOKEN
{%- endcapture -%}

{%- capture python_example -%}
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

{%- capture scala_example -%}
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

{%- capture api_link -%}
[SymmetricDeleteApproach](https://nlp.johnsnowlabs.com/api/com/johnsnowlabs/nlp/annotators/spell/symmetric/SymmetricDeleteApproach)
{%- endcapture -%}

{%- capture python_api_link -%}
[SymmetricDeleteApproach](/api/python/reference/autosummary/python/sparknlp/annotator/spell_check/symmetric_delete/index.html#sparknlp.annotator.spell_check.symmetric_delete.SymmetricDeleteApproach)
{%- endcapture -%}

{%- capture source_link -%}
[SymmetricDeleteApproach](https://github.com/JohnSnowLabs/spark-nlp/tree/master/src/main/scala/com/johnsnowlabs/nlp/annotators/spell/symmetric/SymmetricDeleteApproach.scala)
{%- endcapture -%}

{% include templates/training_anno_template.md
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
