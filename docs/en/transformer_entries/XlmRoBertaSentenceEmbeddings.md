{%- capture title -%}
XlmRoBertaSentenceEmbeddings
{%- endcapture -%}

{%- capture description -%}
Sentence-level embeddings using XLM-RoBERTa. The XLM-RoBERTa model was proposed in [Unsupervised Cross-lingual Representation Learning at Scale](https://arxiv.org/abs/1911.02116)
by Alexis Conneau, Kartikay Khandelwal, Naman Goyal, Vishrav Chaudhary, Guillaume
Wenzek, Francisco GuzmÃ¡n, Edouard Grave, Myle Ott, Luke Zettlemoyer and Veselin Stoyanov. It is based on Facebook's
RoBERTa model released in 2019. It is a large multi-lingual language model, trained on 2.5TB of filtered CommonCrawl
data.

Pretrained models can be loaded with `pretrained` of the companion object:
```
val embeddings = XlmRoBertaSentenceEmbeddings.pretrained()
  .setInputCols("document")
  .setOutputCol("sentence_embeddings")
```
The default model is `"sent_xlm_roberta_base"`, default language is `"xx"` (meaning multi-lingual), if no values are provided.
For available pretrained models please see the [Models Hub](https://nlp.johnsnowlabs.com/models?task=Embeddings).

Models from the HuggingFace 🤗 Transformers library are also compatible with Spark NLP 🚀. To see which models are compatible and how to import them see [Import Transformers into Spark NLP 🚀](https://github.com/JohnSnowLabs/spark-nlp/discussions/5669).

**Paper Abstract:**

*This paper shows that pretraining multilingual language models at scale leads to significant performance gains for a
wide range of cross-lingual transfer tasks. We train a Transformer-based masked language model on one hundred
languages, using more than two terabytes of filtered CommonCrawl data. Our model, dubbed XLM-R, significantly
outperforms multilingual BERT (mBERT) on a variety of cross-lingual benchmarks, including +13.8% average accuracy on
XNLI, +12.3% average F1 score on MLQA, and +2.1% average F1 score on NER. XLM-R performs particularly well on
low-resource languages, improving 11.8% in XNLI accuracy for Swahili and 9.2% for Urdu over the previous XLM model. We
also present a detailed empirical evaluation of the key factors that are required to achieve these gains, including the
trade-offs between (1) positive transfer and capacity dilution and (2) the performance of high and low resource
languages at scale. Finally, we show, for the first time, the possibility of multilingual modeling without sacrificing
per-language performance; XLM-Ris very competitive with strong monolingual models on the GLUE and XNLI benchmarks. We
will make XLM-R code, data, and models publicly available.*

**Tips:**
  - XLM-RoBERTa is a multilingual model trained on 100 different languages. Unlike some XLM multilingual models, it does
    not require **lang** parameter to understand which language is used, and should be able to determine the correct
    language from the input ids.
  - This implementation is the same as RoBERTa. Refer to the RoBertaEmbeddings for usage examples
    as well as the information relative to the inputs and outputs.
{%- endcapture -%}

{%- capture input_anno -%}
DOCUMENT
{%- endcapture -%}

{%- capture output_anno -%}
SENTENCE_EMBEDDINGS
{%- endcapture -%}

{%- capture python_api_link -%}
[XlmRoBertaSentenceEmbeddings](/api/python/reference/autosummary/python/sparknlp/annotator/embeddings/xlm_roberta_sentence_embeddings/index.html#sparknlp.annotator.embeddings.xlm_roberta_sentence_embeddings.XlmRoBertaSentenceEmbeddings)
{%- endcapture -%}

{%- capture api_link -%}
[XlmRoBertaSentenceEmbeddings](https://nlp.johnsnowlabs.com/api/com/johnsnowlabs/nlp/embeddings/XlmRoBertaSentenceEmbeddings)
{%- endcapture -%}

{%- capture source_link -%}
[XlmRoBertaSentenceEmbeddings](https://github.com/JohnSnowLabs/spark-nlp/tree/master/src/main/scala/com/johnsnowlabs/nlp/embeddings/XlmRoBertaSentenceEmbeddings.scala)
{%- endcapture -%}

{%- capture prediction_python_example -%}
# Coming Soon!
{%- endcapture -%}

{%- capture prediction_scala_example -%}
// Coming Soon!
{%- endcapture -%}

{%- capture training_python_example -%}
import sparknlp
from sparknlp.base import *
from sparknlp.annotator import *
from pyspark.ml import Pipeline

smallCorpus = spark.read.option("header","True").csv("sentiment.csv")

documentAssembler = DocumentAssembler() \
    .setInputCol("text") \
    .setOutputCol("document")

embeddings = XlmRoBertaSentenceEmbeddings.pretrained() \
  .setInputCols(["document"])\
  .setOutputCol("sentence_embeddings")

# Then the training can start with the transformer embeddings
docClassifier = ClassifierDLApproach() \
    .setInputCols("sentence_embeddings") \
    .setOutputCol("category") \
    .setLabelColumn("label") \
    .setBatchSize(64) \
    .setMaxEpochs(20) \
    .setLr(5e-3) \
    .setDropout(0.5)

pipeline = Pipeline().setStages([
    documentAssembler,
    embeddings,
    docClassifier
])

pipelineModel = pipeline.fit(smallCorpus)
{%- endcapture -%}

{%- capture training_scala_example -%}
import com.johnsnowlabs.nlp.base.DocumentAssembler
import com.johnsnowlabs.nlp.embeddings.RoBertaSentenceEmbeddings
import com.johnsnowlabs.nlp.annotators.classifier.dl.ClassifierDLApproach
import org.apache.spark.ml.Pipeline

val smallCorpus = spark.read.option("header", "true").csv("src/test/resources/classifier/sentiment.csv")

val documentAssembler = new DocumentAssembler()
  .setInputCol("text")
  .setOutputCol("document")

val embeddings = XlmRoBertaSentenceEmbeddings.pretrained()
  .setInputCols("document")
  .setOutputCol("sentence_embeddings")

// Then the training can start with the transformer embeddings
val docClassifier = new ClassifierDLApproach()
  .setInputCols("sentence_embeddings")
  .setOutputCol("category")
  .setLabelColumn("label")
  .setBatchSize(64)
  .setMaxEpochs(20)
  .setLr(5e-3f)
  .setDropout(0.5f)

val pipeline = new Pipeline().setStages(Array(
  documentAssembler,
  embeddings,
  docClassifier
))

val pipelineModel = pipeline.fit(smallCorpus)
{%- endcapture -%}

{%- capture embeddings_python_example -%}
import sparknlp
from sparknlp.base import *
from sparknlp.common import *
from sparknlp.annotator import *
from sparknlp.training import *
from pyspark.ml import Pipeline

documentAssembler = DocumentAssembler() \
    .setInputCol("text") \
    .setOutputCol("document")

tokenizer = Tokenizer() \
    .setInputCols(["document"]) \
    .setOutputCol("token")

sentenceEmbeddings = XlmRoBertaSentenceEmbeddings.pretrained() \
    .setInputCols(["document"]) \
    .setOutputCol("sentence_embeddings") \
    .setCaseSensitive(True)

# you can either use the output to train ClassifierDL, SentimentDL, or MultiClassifierDL
# or you can use EmbeddingsFinisher to prepare the results for Spark ML functions

embeddingsFinisher = EmbeddingsFinisher() \
    .setInputCols(["sentence_embeddings"]) \
    .setOutputCols("finished_embeddings") \
    .setOutputAsVector(True) \
    .setCleanAnnotations(False)

pipeline = Pipeline() \
    .setStages([
      documentAssembler,
      tokenizer,
      sentenceEmbeddings,
      embeddingsFinisher
    ])

data = spark.createDataFrame([["This is a sentence."]]).toDF("text")
result = pipeline.fit(data).transform(data)

result.selectExpr("explode(finished_embeddings) as result").show(5, 80)
+--------------------------------------------------------------------------------+
|                                                                          result|
+--------------------------------------------------------------------------------+
|[-0.05969233065843582,-0.030789051204919815,0.04443822056055069,0.09564960747...|
|[-0.038839809596538544,0.011712731793522835,0.019954433664679527,0.0667808502...|
|[-0.03952755779027939,-0.03455188870429993,0.019103847444057465,0.04311436787...|
|[-0.09579929709434509,0.02494969218969345,-0.014753809198737144,0.10259044915...|
|[0.004710011184215546,-0.022148698568344116,0.011723337695002556,-0.013356896...|
+--------------------------------------------------------------------------------+

{%- endcapture -%}

{%- capture embeddings_scala_example -%}
import spark.implicits._
import com.johnsnowlabs.nlp.base._
import com.johnsnowlabs.nlp.annotator._
import com.johnsnowlabs.nlp.EmbeddingsFinisher
import org.apache.spark.ml.Pipeline

val documentAssembler = new DocumentAssembler()
  .setInputCol("text")
  .setOutputCol("document")

val tokenizer = new Tokenizer()
  .setInputCols(Array("document"))
  .setOutputCol("token")

val sentenceEmbeddings = XlmRoBertaSentenceEmbeddings.pretrained()
  .setInputCols("document")
  .setOutputCol("sentence_embeddings")
  .setCaseSensitive(true)

// you can either use the output to train ClassifierDL, SentimentDL, or MultiClassifierDL
// or you can use EmbeddingsFinisher to prepare the results for Spark ML functions

val embeddingsFinisher = new EmbeddingsFinisher()
  .setInputCols("sentence_embeddings")
  .setOutputCols("finished_embeddings")
  .setOutputAsVector(true)
  .setCleanAnnotations(false)

val pipeline = new Pipeline()
  .setStages(Array(
    documentAssembler,
    tokenizer,
    sentenceEmbeddings,
    embeddingsFinisher
  ))

val data = Seq("This is a sentence.").toDF("text")
val result = pipeline.fit(data).transform(data)

result.selectExpr("explode(finished_embeddings) as result").show(5, 80)
+--------------------------------------------------------------------------------+
|                                                                          result|
+--------------------------------------------------------------------------------+
|[-0.05969233065843582,-0.030789051204919815,0.04443822056055069,0.09564960747...|
|[-0.038839809596538544,0.011712731793522835,0.019954433664679527,0.0667808502...|
|[-0.03952755779027939,-0.03455188870429993,0.019103847444057465,0.04311436787...|
|[-0.09579929709434509,0.02494969218969345,-0.014753809198737144,0.10259044915...|
|[0.004710011184215546,-0.022148698568344116,0.011723337695002556,-0.013356896...|
+--------------------------------------------------------------------------------+

{%- endcapture -%}

{% include templates/transformer_usecases_template.md
title=title
description=description
input_anno=input_anno
output_anno=output_anno
python_api_link=python_api_link
api_link=api_link
source_link=source_link
prediction_python_example=prediction_python_example
prediction_scala_example=prediction_scala_example
training_python_example=training_python_example
training_scala_example=training_scala_example
embeddings_python_example=embeddings_python_example
embeddings_scala_example=embeddings_scala_example
%}