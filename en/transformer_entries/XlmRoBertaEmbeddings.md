{%- capture title -%}
XlmRoBertaEmbeddings
{%- endcapture -%}

{%- capture description -%}
The XLM-RoBERTa model was proposed in [Unsupervised Cross-lingual Representation Learning at Scale](https://arxiv.org/abs/1911.02116)
by Alexis Conneau, Kartikay Khandelwal, Naman Goyal, Vishrav Chaudhary, Guillaume
Wenzek, Francisco GuzmÃ¡n, Edouard Grave, Myle Ott, Luke Zettlemoyer and Veselin Stoyanov. It is based on Facebook's
RoBERTa model released in 2019. It is a large multi-lingual language model, trained on 2.5TB of filtered CommonCrawl
data.

Pretrained models can be loaded with `pretrained` of the companion object:
```
val embeddings = XlmRoBertaEmbeddings.pretrained()
  .setInputCols("document", "token")
  .setOutputCol("embeddings")
```
The default model is `"xlm_roberta_base"`, default language is `"xx"` (meaning multi-lingual), if no values are provided.
For available pretrained models please see the [Models Hub](https://nlp.johnsnowlabs.com/models?task=Embeddings).

For extended examples of usage, see the [Spark NLP Workshop](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/jupyter/transformers/HuggingFace%20in%20Spark%20NLP%20-%20XLM-RoBERTa.ipynb)
and the [XlmRoBertaEmbeddingsTestSpec](https://github.com/JohnSnowLabs/spark-nlp/tree/master/src/test/scala/com/johnsnowlabs/nlp/embeddings/XlmRoBertaEmbeddingsTestSpec.scala).
Models from the HuggingFace 🤗 Transformers library are also compatible with Spark NLP 🚀. The Spark NLP Workshop
example shows how to import them.

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
DOCUMENT, TOKEN
{%- endcapture -%}

{%- capture output_anno -%}
WORD_EMBEDDINGS
{%- endcapture -%}

{%- capture python_example -%}
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

embeddings = XlmRoBertaEmbeddings.pretrained() \
    .setInputCols(["document", "token"]) \
    .setOutputCol("embeddings") \
    .setCaseSensitive(True)

embeddingsFinisher = EmbeddingsFinisher() \
    .setInputCols(["embeddings"]) \
    .setOutputCols("finished_embeddings") \
    .setOutputAsVector(True) \
    .setCleanAnnotations(False)

pipeline = Pipeline() \
    .setStages([
      documentAssembler,
      tokenizer,
      embeddings,
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

{%- capture scala_example -%}
import spark.implicits._
import com.johnsnowlabs.nlp.base.DocumentAssembler
import com.johnsnowlabs.nlp.annotators.Tokenizer
import com.johnsnowlabs.nlp.embeddings.XlmRoBertaEmbeddings
import com.johnsnowlabs.nlp.EmbeddingsFinisher
import org.apache.spark.ml.Pipeline

val documentAssembler = new DocumentAssembler()
  .setInputCol("text")
  .setOutputCol("document")

val tokenizer = new Tokenizer()
  .setInputCols(Array("document"))
  .setOutputCol("token")

val embeddings = XlmRoBertaEmbeddings.pretrained()
  .setInputCols("document", "token")
  .setOutputCol("embeddings")
  .setCaseSensitive(true)

val embeddingsFinisher = new EmbeddingsFinisher()
  .setInputCols("embeddings")
  .setOutputCols("finished_embeddings")
  .setOutputAsVector(true)
  .setCleanAnnotations(false)

val pipeline = new Pipeline()
  .setStages(Array(
    documentAssembler,
    tokenizer,
    embeddings,
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

{%- capture api_link -%}
[XlmRoBertaEmbeddings](https://nlp.johnsnowlabs.com/api/com/johnsnowlabs/nlp/embeddings/XlmRoBertaEmbeddings)
{%- endcapture -%}

{%- capture source_link -%}
[XlmRoBertaEmbeddings](https://github.com/JohnSnowLabs/spark-nlp/tree/master/src/main/scala/com/johnsnowlabs/nlp/embeddings/XlmRoBertaEmbeddings.scala)
{%- endcapture -%}

{% include templates/anno_template.md
title=title
description=description
input_anno=input_anno
output_anno=output_anno
python_example=python_example
scala_example=scala_example
api_link=api_link
source_link=source_link
%}