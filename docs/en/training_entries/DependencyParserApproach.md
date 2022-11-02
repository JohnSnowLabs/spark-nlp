{%- capture title -%}
DependencyParserApproach
{%- endcapture -%}

{%- capture description -%}
Trains an unlabeled parser that finds a grammatical relations between two words in a sentence.

Dependency parser provides information about word relationship. For example, dependency parsing can tell you what
the subjects and objects of a verb are, as well as which words are modifying (describing) the subject. This can help
you find precise answers to specific questions.

The required training data can be set in two different ways (only one can be chosen for a particular model):
  - Dependency treebank in the [Penn Treebank format](http://www.nltk.org/nltk_data/) set with `setDependencyTreeBank`. Data Format:
    ```
    (S
    (S-TPC-1
      (NP-SBJ
        (NP (NP (DT A) (NN form)) (PP (IN of) (NP (NN asbestos))))
        (RRC ...)...)...)
    ...
    (VP (VBD reported) (SBAR (-NONE- 0) (S (-NONE- *T*-1))))
    (. .))
    ```
  - Dataset in the [CoNLL-U format](https://universaldependencies.org/format.html) set with `setConllU`.
    Data Format:
    ```
    -DOCSTART- -X- -X- O

    EU NNP B-NP B-ORG
    rejects VBZ B-VP O
    German JJ B-NP B-MISC
    ```

Apart from that, no additional training data is needed.

See [DependencyParserApproachTestSpec](https://github.com/JohnSnowLabs/spark-nlp/blob/master/src/test/scala/com/johnsnowlabs/nlp/annotators/parser/dep/DependencyParserApproachTestSpec.scala) for further reference on how to use this API.
{%- endcapture -%}

{%- capture input_anno -%}
DOCUMENT, POS, TOKEN
{%- endcapture -%}

{%- capture output_anno -%}
DEPENDENCY
{%- endcapture -%}

{%- capture python_example -%}
import sparknlp
from sparknlp.base import *
from sparknlp.annotator import *
from pyspark.ml import Pipeline

documentAssembler = DocumentAssembler() \
    .setInputCol("text") \
    .setOutputCol("document")

sentence = SentenceDetector() \
    .setInputCols("document") \
    .setOutputCol("sentence")

tokenizer = Tokenizer() \
    .setInputCols("sentence") \
    .setOutputCol("token")

posTagger = PerceptronModel.pretrained() \
    .setInputCols("sentence", "token") \
    .setOutputCol("pos")

dependencyParserApproach = DependencyParserApproach() \
    .setInputCols("sentence", "pos", "token") \
    .setOutputCol("dependency") \
    .setDependencyTreeBank("src/test/resources/parser/unlabeled/dependency_treebank")

pipeline = Pipeline().setStages([
    documentAssembler,
    sentence,
    tokenizer,
    posTagger,
    dependencyParserApproach
])

# Additional training data is not needed, the dependency parser relies on the dependency tree bank / CoNLL-U only.
emptyDataSet = spark.createDataFrame([[""]]).toDF("text")
pipelineModel = pipeline.fit(emptyDataSet)

{%- endcapture -%}

{%- capture scala_example -%}
import spark.implicits._
import com.johnsnowlabs.nlp.base.DocumentAssembler
import com.johnsnowlabs.nlp.annotators.sbd.pragmatic.SentenceDetector
import com.johnsnowlabs.nlp.annotators.Tokenizer
import com.johnsnowlabs.nlp.annotators.pos.perceptron.PerceptronModel
import com.johnsnowlabs.nlp.annotators.parser.dep.DependencyParserApproach
import org.apache.spark.ml.Pipeline

val documentAssembler = new DocumentAssembler()
  .setInputCol("text")
  .setOutputCol("document")

val sentence = new SentenceDetector()
  .setInputCols("document")
  .setOutputCol("sentence")

val tokenizer = new Tokenizer()
  .setInputCols("sentence")
  .setOutputCol("token")

val posTagger = PerceptronModel.pretrained()
  .setInputCols("sentence", "token")
  .setOutputCol("pos")

val dependencyParserApproach = new DependencyParserApproach()
  .setInputCols("sentence", "pos", "token")
  .setOutputCol("dependency")
  .setDependencyTreeBank("src/test/resources/parser/unlabeled/dependency_treebank")

val pipeline = new Pipeline().setStages(Array(
  documentAssembler,
  sentence,
  tokenizer,
  posTagger,
  dependencyParserApproach
))

// Additional training data is not needed, the dependency parser relies on the dependency tree bank / CoNLL-U only.
val emptyDataSet = Seq.empty[String].toDF("text")
val pipelineModel = pipeline.fit(emptyDataSet)

{%- endcapture -%}

{%- capture api_link -%}
[DependencyParserApproach](https://nlp.johnsnowlabs.com/api/com/johnsnowlabs/nlp/annotators/parser/dep/DependencyParserApproach)
{%- endcapture -%}

{%- capture python_api_link -%}
[DependencyParserApproach](/api/python/reference/autosummary/python/sparknlp/annotator/dependency/dependency_parser/index.html#sparknlp.annotator.dependency.dependency_parser.DependencyParserApproach)
{%- endcapture -%}

{%- capture source_link -%}
[DependencyParserApproach](https://github.com/JohnSnowLabs/spark-nlp/tree/master/src/main/scala/com/johnsnowlabs/nlp/annotators/parser/dep/DependencyParserApproach.scala)
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
