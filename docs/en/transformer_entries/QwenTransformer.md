{%- capture title -%} QwenTransformer {%- endcapture -%} 
{%- capture description -%} 
Qwen is a family of decoder-only large language models developed by the Qwen team (Alibaba) and optimized for advanced reasoning, code generation, chat and general NLP tasks. The series includes multiple sizes (from ~0.5B up to 72B parameters) with stable support for long contexts (commonly 32K tokens for released models) and specialized variants for chat, code, and multimodal workloads. 

Pretrained models can be loaded with the `pretrained` method of the companion object:

```scala
val qwen = QwenTransformer.pretrained()
     .setInputCols("document")
     .setOutputCol("generation")
```

The default model is `"qwen_7.5b_chat"`, if no name is provided. For available pretrained models please see the Models Hub. 

For available pretrained models please see the [Models Hub](https://sparknlp.org/models?annotator=Phi3Transformer).

Spark NLP also supports Hugging Face transformer-based translation models. Learn more here:  
- [Import models into Spark NLP](https://github.com/JohnSnowLabs/spark-nlp/discussions/5669)

**Resources**:
- [Qwen official site (Qwen Chat)](https://qwen.ai/)  
- [Qwen GitHub (Qwen / Qwen)](https://github.com/QwenLM/Qwen)  
- [Qwen on Hugging Face (organization)](https://huggingface.co/Qwen)  

**Paper abstract**

*In this report, we introduce Qwen2.5, a comprehensive series of large language models (LLMs) designed to meet diverse needs. Compared to previous iterations, Qwen 2.5 has been significantly improved during both the pre-training and post-training stages. In terms of pre-training, we have scaled the high-quality pre-training datasets from the previous 7 trillion tokens to 18 trillion tokens. This provides a strong foundation for common sense, expert knowledge, and reasoning capabilities. In terms of post-training, we implement intricate supervised finetuning with over 1 million samples, as well as multistage reinforcement learning. Post-training techniques enhance human preference, and notably improve long text generation, structural data analysis, and instruction following. To handle diverse and varied use cases effectively, we present Qwen2.5 LLM series in rich sizes. Open-weight offerings include base and instruction-tuned models, with quantized versions available. In addition, for hosted solutions, the proprietary models currently include two mixture-of-experts (MoE) variants: Qwen2.5-Turbo and Qwen2.5-Plus, both available from Alibaba Cloud Model Studio. Qwen2.5 has demonstrated top-tier performance on a wide range of benchmarks evaluating language understanding, reasoning, mathematics, coding, human preference alignment, etc. Specifically, the open-weight flagship Qwen2.5-72B-Instruct outperforms a number of open and proprietary models and demonstrates competitive performance to the state-of-the-art open-weight model, Llama-3-405B-Instruct, which is around 5 times larger. Qwen2.5-Turbo and Qwen2.5-Plus offer superior cost-effectiveness while performing competitively against GPT-4o-mini and GPT-4o respectively. Additionally, as the foundation, Qwen2.5 models have been instrumental in training specialized models such as Qwen2.5-Math, Qwen2.5-Coder, QwQ, and multimodal models.*

{%- endcapture -%}

{%- capture input_anno -%}
DOCUMENT
{%- endcapture -%}

{%- capture output_anno -%}
GENERATION
{%- endcapture -%}

{%- capture api_link -%}
[QwenTransformer](/api/com/johnsnowlabs/nlp/annotators/seq2seq/QwenTransformer.html)
{%- endcapture -%}

{%- capture python_api_link -%}
[QwenTransformer](/api/python/reference/autosummary/sparknlp/annotator/seq2seq/qwen_transformer/index.html)
{%- endcapture -%}

{%- capture source_link -%}
[QwenTransformer](https://github.com/JohnSnowLabs/spark-nlp/blob/master/src/main/scala/com/johnsnowlabs/nlp/annotators/seq2seq/QwenTransformer.scala)
{%- endcapture -%}

{%- capture python_example -%}
from sparknlp.base import DocumentAssembler
from sparknlp.annotator import QwenTransformer
from pyspark.ml import Pipeline

documentAssembler = DocumentAssembler() \
  .setInputCol("text") \
  .setOutputCol("document")   

qwen = QwenTransformer.pretrained("qwen_7.5b_chat") \
  .setInputCols(["document"]) \
  .setMaxOutputLength(100) \
  .setOutputCol("generation")

pipeline = Pipeline().setStages([
    documentAssembler,
    qwen
])

data = spark.createDataFrame([
    ["What is 7 × 8?"]
]).toDF("text")

model = pipeline.fit(data)
result = model.transform(data)

result.select("generation.result").show(truncate=False)

+--------------------------------+
|result                          |
+--------------------------------+
|[The product of 7 and 8 is 56.] |
+--------------------------------+
{%- endcapture -%}

{%- capture scala_example -%}
import com.johnsnowlabs.nlp.base._
import com.johnsnowlabs.nlp.annotator._
import org.apache.spark.ml.Pipeline

val documentAssembler = new DocumentAssembler()
  .setInputCol("text")
  .setOutputCol("document")

val qwen = QwenTransformer.pretrained("qwen_7.5b_chat")
  .setInputCols("document")
  .setMaxOutputLength(100)
  .setOutputCol("generation")

val pipeline = new Pipeline().setStages(Array(
  documentAssembler,
  qwen
))

val data = Seq("What is 7 × 8?").toDF("text")

val model = pipeline.fit(data)
val result = model.transform(data)

result.select("generation.result").show(false)

+--------------------------------+
|result                          |
+--------------------------------+
|[The product of 7 and 8 is 56.] |
+--------------------------------+
{%- endcapture -%}

{%- capture resources -%}
**Resources**:
- [Qwen official site (Qwen Chat)](https://qwen.ai/)  
- [Qwen GitHub (Qwen / Qwen)](https://github.com/QwenLM/Qwen)  
- [Qwen on Hugging Face (organization)](https://huggingface.co/Qwen)  
- [Qwen technical report / papers (arXiv)](https://arxiv.org/search/?query=Qwen&searchtype=all)  
- [Spark NLP QwenTransformer docs](https://sparknlp.org/api/python/reference/autosummary/sparknlp/annotator/seq2seq/qwen_transformer/index.html)  
{%- endcapture -%}

{%- capture paper_abstract -%}
*We introduce Qwen, a comprehensive language model series consisting of base and chat models across multiple sizes. The series demonstrates strong performance on a wide range of downstream tasks, with chat variants aligned via human feedback and specialized models for coding and math. Models support extended context lengths (commonly up to 32K tokens) and show competitive results compared to contemporaneous open-source models.*  
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
