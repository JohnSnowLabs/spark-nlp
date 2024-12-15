---
layout: docs
header: true
seotitle:
title: Automatic Speech Recognition
permalink: docs/en/tasks/automatic_speech_recognition
key: docs-tasks-automatic-speech-recognition
modify_date: "2024-09-26"
show_nav: true
sidebar:
  nav: sparknlp
---

**Automatic Speech Recognition (ASR)** is the technology that enables computers to recognize and process human speech into text. ASR plays a vital role in numerous applications, from voice-activated assistants to transcription services, making it an essential part of modern natural language processing (NLP) solutions. Spark NLP provides powerful tools for implementing ASR systems effectively.

In this context, ASR involves converting spoken language into text by analyzing audio signals. Common use cases include:

- **Voice Assistants:** Enabling devices like smartphones and smart speakers to understand and respond to user commands.
- **Transcription Services:** Automatically converting audio recordings from meetings, interviews, or lectures into written text.
- **Accessibility:** Helping individuals with disabilities interact with technology through voice commands.

By leveraging ASR, organizations can enhance user experience, improve accessibility, and streamline workflows that involve audio data.

<div style="text-align: center;">
  <iframe width="560" height="315" src="https://www.youtube.com/embed/hscJK-4kA_A?si=xYgz6ejc-2XvXcoR" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>
</div>

## Picking a Model

When selecting a model for Automatic Speech Recognition, it’s essential to evaluate several factors to ensure optimal performance for your specific use case. Begin by analyzing the **nature of your audio data**, considering the accent, language, and quality of the recordings. Determine if your task requires **real-time transcription** or if batch processing is sufficient, as some models excel in specific scenarios.

Next, assess the **model complexity**; simpler models may suffice for straightforward tasks, while more sophisticated models are better suited for nuanced speech recognition. Consider the **availability of diverse audio data** for training, as larger datasets can significantly enhance model performance. Define key **performance metrics** (e.g., word error rate, accuracy) to guide your choice, and ensure the model's interpretability meets your requirements. Finally, account for **resource constraints**, as advanced models typically demand more memory and processing power.

To explore and select from a variety of models, visit [Spark NLP Models](https://sparknlp.org/models), where you can find models tailored for different ASR tasks and languages.

#### Recommended Models for Automatic Speech Recognition Tasks
- **General Speech Recognition:** Use models like [`asr_wav2vec2_large_xlsr_53_english_by_jonatasgrosman`](https://sparknlp.org/2022/09/24/asr_wav2vec2_large_xlsr_53_english_by_jonatasgrosman_en.html){:target="_blank"} for general-purpose transcription.
- **Multilingual Support:** For applications requiring support for multiple languages, consider using models like [`asr_wav2vec2_large_xlsr_53_portuguese_by_jonatasgrosman`](https://sparknlp.org/2021/12/15/wav2vec2.html){:target="_blank"} from the [`Wav2Vec2ForCTC`](https://sparknlp.org/docs/en/transformers#wav2vec2forctc){:target="_blank"} transformer.

By thoughtfully considering these factors and using the right models, you can enhance your ASR applications significantly.

## How to use

<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPython.html %}
```python
import sparknlp
from sparknlp.base import *
from sparknlp.annotator import *
from pyspark.ml import Pipeline

# Step 1: Assemble the raw audio content into a suitable format
audioAssembler = AudioAssembler() \
    .setInputCol("audio_content") \
    .setOutputCol("audio_assembler")

# Step 2: Load a pre-trained Wav2Vec2 model for automatic speech recognition (ASR)
speechToText = Wav2Vec2ForCTC \
    .pretrained() \
    .setInputCols(["audio_assembler"]) \
    .setOutputCol("text")

# Step 3: Define the pipeline with audio assembler and speech-to-text model
pipeline = Pipeline().setStages([audioAssembler, speechToText])

# Step 4: Create a DataFrame containing the raw audio content (as floats)
processedAudioFloats = spark.createDataFrame([[rawFloats]]).toDF("audio_content")

# Step 5: Fit the pipeline and transform the audio data
result = pipeline.fit(processedAudioFloats).transform(processedAudioFloats)

# Step 6: Display the transcribed text from the audio
result.select("text.result").show(truncate = False)

+------------------------------------------------------------------------------------------+
|result                                                                                    |
+------------------------------------------------------------------------------------------+
|[MISTER QUILTER IS THE APOSTLE OF THE MIDLE CLASES AND WE ARE GLAD TO WELCOME HIS GOSPEL ]|
+------------------------------------------------------------------------------------------+
```
```scala
import spark.implicits._
import com.johnsnowlabs.nlp.base._
import com.johnsnowlabs.nlp.annotators._
import com.johnsnowlabs.nlp.annotators.audio.Wav2Vec2ForCTC
import org.apache.spark.ml.Pipeline

// Step 1: Assemble the raw audio content into a suitable format
val audioAssembler: AudioAssembler = new AudioAssembler()
  .setInputCol("audio_content")
  .setOutputCol("audio_assembler")

// Step 2: Load a pre-trained Wav2Vec2 model for automatic speech recognition (ASR)
val speechToText: Wav2Vec2ForCTC = Wav2Vec2ForCTC
  .pretrained()
  .setInputCols("audio_assembler")
  .setOutputCol("text")

// Step 3: Define the pipeline with audio assembler and speech-to-text model
val pipeline: Pipeline = new Pipeline().setStages(Array(audioAssembler, speechToText))

// Step 4: Load raw audio floats from a CSV file
val bufferedSource =
  scala.io.Source.fromFile("src/test/resources/audio/csv/audio_floats.csv")

// Step 5: Extract raw audio floats from CSV and convert to an array of floats
val rawFloats = bufferedSource
  .getLines()
  .map(_.split(",").head.trim.toFloat)
  .toArray
bufferedSource.close

// Step 6: Create a DataFrame with raw audio content (as floats)
val processedAudioFloats = Seq(rawFloats).toDF("audio_content")

// Step 7: Fit the pipeline and transform the audio data
val result = pipeline.fit(processedAudioFloats).transform(processedAudioFloats)

// Step 8: Display the transcribed text from the audio
result.select("text.result").show(truncate = false)

+------------------------------------------------------------------------------------------+
|result                                                                                    |
+------------------------------------------------------------------------------------------+
|[MISTER QUILTER IS THE APOSTLE OF THE MIDLE CLASES AND WE ARE GLAD TO WELCOME HIS GOSPEL ]|
+------------------------------------------------------------------------------------------+
```
</div>

## Try Real-Time Demos!

If you want to see the outputs of ASR models in real time, visit our interactive demos:

- **[Wav2Vec2ForCTC](https://huggingface.co/spaces/abdullahmubeen10/sparknlp-Wav2Vec2ForCTC){:target="_blank"}** – Try this powerful model for real-time speech-to-text from raw audio.
- **[WhisperForCTC](https://huggingface.co/spaces/abdullahmubeen10/sparknlp-WhisperForCTC){:target="_blank"}** – Test speech recognition in multiple languages and noisy environments.
- **[HubertForCTC](https://huggingface.co/spaces/abdullahmubeen10/sparknlp-HubertForCTC){:target="_blank"}** – Experience quick and accurate voice command recognition.

## Useful Resources

Want to dive deeper into Automatic Speech Recognition with Spark NLP? Here are somText Preprocessinge curated resources to help you get started and explore further:

**Articles and Guides**
- *[Converting Speech to Text with Spark NLP and Python](https://www.johnsnowlabs.com/converting-speech-to-text-with-spark-nlp-and-python/){:target="_blank"}*
- *[Simplify Your Speech Recognition Workflow with SparkNLP](https://medium.com/spark-nlp/simplify-your-speech-recognition-workflow-with-sparknlp-e381606e4e82){:target="_blank"}*
- *[Vision Transformers and Automatic Speech Recognition in Spark NLP](https://www.nlpsummit.org/vision-transformers-and-automatic-speech-recognition-in-spark-nlp/){:target="_blank"}*

**Notebooks**
- *[Automatic Speech Recognition in Spark NLP](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Public/17.Speech_Recognition.ipynb){:target="_blank"}*
- *[Speech Recognition Transformers in Spark NLP](https://github.com/JohnSnowLabs/spark-nlp/tree/master/examples/python/annotation/audio){:target="_blank"}*
