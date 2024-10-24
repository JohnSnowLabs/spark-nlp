---
layout: docs
header: true
seotitle:
title: Zero-Shot Classification
permalink: docs/en/tasks/zero_shot_classification
key: docs-tasks-zero_shot_classification
modify_date: "2024-09-26"
show_nav: true
sidebar:
  nav: sparknlp
---

**Zero-Shot Classification** is a method of classifying unseen labels in text without needing any prior training data for those labels. This technique is especially useful for scenarios where pre-defined categories are not available, allowing for flexibility in categorizing text based on descriptions of labels alone. Spark NLP offers state-of-the-art solutions for zero-shot classification, enabling users to classify texts into various categories even when no labeled data is available.

Zero-shot classification processes text at a broader level, where the system predicts the most relevant labels based on their descriptions. Typical use cases include:

- **Text Categorization:** Automatically classifying text into a set of predefined or custom categories based on label descriptions.

By leveraging zero-shot classification, organizations can classify large volumes of text data without the need to curate annotated datasets for each possible label, significantly reducing manual efforts in text annotation and data preparation.

## Picking a Model

When selecting a model for zero-shot classification, it is important to consider several factors that impact performance. First, analyze the **range of labels or categories** you want to classify. Zero-shot classification is versatile, but choosing models trained on broader datasets often yields better results.

Next, consider the **complexity of your text**. Is it formal or informal? Does it involve domain-specific language such as legal or healthcare text? **Performance metrics** (e.g., accuracy, precision, recall) help assess whether a model fits your requirements. Additionally, ensure you evaluate your **computational resources**, as larger models, like those based on transformer architectures, may require significant memory and processing power.

You can explore and select models for your zero-shot classification tasks at [Spark NLP Models](https://sparknlp.org/models), where you'll find a variety of models for specific datasets and classification challenges.

#### Recommended Models for Zero-Shot Classification

- **Zero-Shot Text Classification:** Consider using models like [`bart-large-mnli`](https://sparknlp.org/2024/08/27/bart_large_zero_shot_classifier_mnli_en.html){:target="_blank"} for general-purpose multilingual text data classification across various domains.
- **Zero-Shot Named Entity Recognition (NER):** Use models like [`zero_shot_ner_roberta`](https://sparknlp.org/2023/02/08/zero_shot_ner_roberta_en.html){:target="_blank"} for identifying entities across various domains and languages without requiring task-specific labeled data.

If pre-trained models don't match your exact needs, you can train your own custom model using the [Spark NLP Training Documentation](https://sparknlp.org/docs/en/training).

By selecting the appropriate zero-shot classification model, you can expand your ability to analyze text data without predefined labels, providing flexibility for dynamic and evolving classification tasks.

## How to use

<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPython.html %}
```python
import sparknlp
from sparknlp.base import *
from sparknlp.annotator import *
from pyspark.ml import Pipeline

# 1. Document Assembler: Converts raw input text into a document format.
documentAssembler = DocumentAssembler() \
    .setInputCol("text") \
    .setOutputCol("document")

# 2. Tokenizer: Splits the document into individual tokens (words).
tokenizer = Tokenizer() \
    .setInputCols(["document"]) \
    .setOutputCol("token")

# 3. Pre-trained Sequence Classifier (Zero-Shot Classification): Loads a pre-trained BART model for zero-shot classification.
sequenceClassifier = BartForZeroShotClassification.pretrained() \
    .setInputCols(["token", "document"]) \
    .setOutputCol("label") \
    .setCaseSensitive(True)

# 4. Pipeline: Defines a pipeline with three stages - document assembler, tokenizer, and zero-shot classifier.
pipeline = Pipeline().setStages([
    documentAssembler,
    tokenizer,
    sequenceClassifier
])

# 5. Sample Data: Creating a DataFrame with sample text data to test zero-shot classification.
data = spark.createDataFrame([["I loved this movie when I was a child.", "It was pretty boring."]]).toDF("text")

# 6. Fit and Transform: Fits the pipeline to the data and applies the model for classification.
result = pipeline.fit(data).transform(data)

# 7. Displaying Results: Shows the classification labels assigned to each text (e.g., positive or negative sentiment).
result.select("label.result").show(truncate=False)

<!-- Sample Output:
+------+
|result|
+------+
|[pos] |
|[neg] |
+------+ 
-->
```

```scala
import spark.implicits._
import com.johnsnowlabs.nlp.base._
import com.johnsnowlabs.nlp.annotator._
import org.apache.spark.ml.Pipeline

// Assembling the document from the input text
val documentAssembler = new DocumentAssembler()
  .setInputCol("text")
  .setOutputCol("document")

// Tokenizing the text
val tokenizer = new Tokenizer()
  .setInputCols("document")
  .setOutputCol("token")

// Loading the pre-trained zero-shot classification model (BERT)
val sequenceClassifier = BertForZeroShotClassification.pretrained()
  .setInputCols("token", "document")
  .setOutputCol("label")
  .setCaseSensitive(true)

// Creating a pipeline with document assembler, tokenizer, and classifier
val pipeline = new Pipeline().setStages(Array(
  documentAssembler,
  tokenizer,
  sequenceClassifier
))

// Creating a sample DataFrame
val data = Seq("I loved this movie when I was a child.", "It was pretty boring.").toDF("text")

// Fitting the pipeline and transforming the data
val result = pipeline.fit(data).transform(data)

// Showing the results
result.select("label.result").show(false)

// Sample Output:
// +------+
// |result|
// +------+
// |[pos] |
// |[neg] |
// +------+
```
</div>

## Try Real-Time Demos!

If you want to see the outputs of text classification models in real time, visit our interactive demos:

- **[BERT Annotators Demo](https://huggingface.co/spaces/abdullahmubeen10/sparknlp-bert-annotators){:target="_blank"}** – A live demo where you can try your labels and inputs on zero shot classification models on the go.
- **[Zero-Shot Named Entity Recognition (NER)](https://huggingface.co/spaces/abdullahmubeen10/sparknlp-Zero-Shot-NER){:target="_blank"}** – A live demo where you can try your labels and inputs on zero shot classification models on the go.

## Useful Resources

Want to dive deeper into text classification with Spark NLP? Here are some curated resources to help you get started and explore further:

**Notebooks**
- *[Zero-Shot Text Classification in Spark NLP](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Public/5.4_ZeroShot_Text_Classification.ipynb){:target="_blank"}*
- *[Zero-Shot for Named Entity Recognition](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Public/4.2_ZeroShot_NER.ipynb){:target="_blank"}*
