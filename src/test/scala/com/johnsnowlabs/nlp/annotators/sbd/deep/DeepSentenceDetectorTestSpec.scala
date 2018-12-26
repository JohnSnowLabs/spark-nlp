package com.johnsnowlabs.nlp.annotators.sbd.deep

import com.johnsnowlabs.nlp.AnnotatorType.{CHUNK, DOCUMENT, TOKEN}
import com.johnsnowlabs.nlp.SparkAccessor.spark.implicits._
import com.johnsnowlabs.nlp._
import com.johnsnowlabs.nlp.annotator.{NerConverter, Tokenizer}
import com.johnsnowlabs.nlp.annotators.ner.dl.NerDLApproach
import com.johnsnowlabs.nlp.embeddings.WordEmbeddingsFormat

import org.scalatest.FlatSpec


class DeepSentenceDetectorTestSpec extends FlatSpec with DeepSentenceDetectorBehaviors {

  private val documentAssembler = new DocumentAssembler()
    .setInputCol("text")
    .setOutputCol("document")

  private val tokenizer = new Tokenizer()
    .setInputCols(Array("document"))
    .setOutputCol("token")

  private val nerTagger = new NerDLApproach()
    .setInputCols("document", "token")
    .setLabelColumn("label")
    .setOutputCol("ner")
    .setMaxEpochs(400)
    .setPo(0.01f)
    .setLr(0.1f)
    .setBatchSize(15)
    .setEmbeddingsSource("src/test/resources/ner-corpus/embeddings.100d.test.txt",
      100, WordEmbeddingsFormat.TEXT)
    .setExternalDataset("src/test/resources/ner-corpus/sentence-detector/mix_dataset.txt")
    .setRandomSeed(0)

  private val nerTaggerSmallEpochs = new NerDLApproach()
    .setInputCols("document", "token")
    .setLabelColumn("label")
    .setOutputCol("ner")
    .setMaxEpochs(100)
    .setPo(0.01f)
    .setLr(0.1f)
    .setBatchSize(9)
    .setEmbeddingsSource("src/test/resources/ner-corpus/embeddings.100d.test.txt",
      100, WordEmbeddingsFormat.TEXT)
    .setExternalDataset("src/test/resources/ner-corpus/sentence-detector/unpunctuated_dataset.txt")
    .setRandomSeed(0)

  private val nerConverter = new NerConverter()
    .setInputCols(Array("document", "token", "ner"))
    .setOutputCol("ner_con")

  private val deepSentenceDetector = new DeepSentenceDetector()
    .setInputCols(Array("document", "token", "ner_con"))
    .setOutputCol("seg_sentence")
    .setIncludeRules(false)

  private val deepSentenceDetectorSmallEpochs = new DeepSentenceDetector()
    .setInputCols(Array("document", "token", "ner_con"))
    .setOutputCol("seg_sentence")
    .setIncludeRules(true)

  private val finisher = new Finisher()
    .setInputCols("seg_sentence")
    .setOutputCols("sentence")

  private val pipeline = new RecursivePipeline().setStages(
    Array(documentAssembler,
      tokenizer,
      nerTagger,
      nerConverter,
      deepSentenceDetector,
      finisher
    ))
  private val pipelineSmallEpochs = new RecursivePipeline().setStages(
    Array(documentAssembler,
      tokenizer,
      nerTaggerSmallEpochs,
      nerConverter,
      deepSentenceDetectorSmallEpochs,
      finisher
    ))

  private val annotations = Seq(
    Annotation(DOCUMENT, 0, 27, "I am Batman I live in Gotham", Map()),
    Annotation(TOKEN, 0, 0, "I", Map("sentence"->"1")),
    Annotation(TOKEN, 2, 3, "am", Map("sentence"->"1")),
    Annotation(TOKEN, 5, 10, "Batman", Map("sentence"->"1")),
    Annotation(TOKEN, 12, 12, "I", Map("sentence"->"1")),
    Annotation(TOKEN, 14, 17, "live", Map("sentence"->"1")),
    Annotation(TOKEN, 19, 20, "in", Map("sentence"->"1")),
    Annotation(TOKEN, 22, 27, "Gotham", Map("sentence"->"1")),
    Annotation(CHUNK, 0, 0, "I", Map("entity"->"sent")),
    Annotation(CHUNK, 12, 12, "I", Map("entity"->"sent"))
  )

  "A Deep Sentence Detector" should "retrieve NER entities from annotations" ignore {

    val expectedEntities = Seq(Annotation(CHUNK, 0, 0, "I", Map("entity"->"sent")),
                               Annotation(CHUNK, 12, 12, "I", Map("entity"->"sent")))

    val entities = deepSentenceDetector.getNerEntities(annotations)

    assert(entities == expectedEntities)

  }

  it should "retrieve a sentence from annotations" ignore {

    val expectedSentence = "I am Batman I live in Gotham"

    val sentence = deepSentenceDetector.retrieveSentence(annotations)

    assert(sentence == expectedSentence)

  }

  it should "segment sentences" ignore {
    val entities = deepSentenceDetector.getNerEntities(annotations)
    val sentence = deepSentenceDetector.retrieveSentence(annotations)
    val expectedSentence1 = "I am Batman"
    val expectedSentence2 = "I live in Gotham"
    val expectedSegmentedSentences =
      Seq(Annotation(DOCUMENT, 0, expectedSentence1.length-1, expectedSentence1, Map("sentence"->"1")),
          Annotation(DOCUMENT, 0, expectedSentence2.length-1, expectedSentence2, Map("sentence"->"2")))

    val segmentedSentence = deepSentenceDetector.segmentSentence(entities, sentence)

    assert(segmentedSentence == expectedSegmentedSentences)

  }

  it should "retrieve valid NER  entities from punctuated and unpunctuated sentences" in {
//    val testDataSet = Seq("This is a sentence. I love deep learning Winter is coming").toDS.toDF("text")
//
//    val pipeline = new RecursivePipeline().setStages(
//      Array(documentAssembler,
//        tokenizer,
//        nerTagger,
//        nerConverter,
//        deepSentenceDetectorSmallEpochs
//      ))
//
//    pipeline.fit(testDataSet).transform(testDataSet).collect()

    val annotations = Seq(
      Annotation(DOCUMENT, 0, 56, "This is a sentence. I love deep learning Winter is coming", Map()),
      Annotation(TOKEN, 0, 3, "This", Map("sentence"->"1")),
      Annotation(TOKEN, 5, 6, "is", Map("sentence"->"1")),
      Annotation(TOKEN, 8, 8, "a", Map("sentence"->"1")),
      Annotation(TOKEN, 10, 17, "sentence", Map("sentence"->"1")),
      Annotation(TOKEN, 18, 18, ".", Map("sentence"->"1")),
      Annotation(TOKEN, 20, 20, "I", Map("sentence"->"1")),
      Annotation(TOKEN, 22, 25, "love", Map("sentence"->"1")),
      Annotation(TOKEN, 27, 30, "deep", Map("sentence"->"1")),
      Annotation(TOKEN, 32, 39, "learning", Map("sentence"->"1")),
      Annotation(TOKEN, 41, 46, "Winter", Map("sentence"->"1")),
      Annotation(TOKEN, 48, 49, "is", Map("sentence"->"1")),
      Annotation(TOKEN, 51, 56, "coming", Map("sentence"->"1")),
      Annotation(CHUNK, 0, 3, "This", Map("entity"->"sent")),
      Annotation(CHUNK, 5, 6, "is", Map("entity"->"sent")),
      Annotation(CHUNK, 20, 20, "I", Map("entity"->"sent")),
      Annotation(CHUNK, 41, 46, "Winter", Map("entity"->"sent"))
    )

    val pragmaticSentenceDetector = Seq(
      Annotation(DOCUMENT, 0, 18, "This is a sentence.", Map()),
      Annotation(DOCUMENT, 20, 56, "I love deep learning Winter is coming", Map())
    )

    val expectedResult = Seq(
      Annotation(DOCUMENT, 0, 19, "This is a sentence.", Map()),
      Annotation(DOCUMENT, 0, 36, "I love deep learning Winter is coming", Map())
    )

    val result = deepSentenceDetector.retrieveValidNerEntities(annotations, pragmaticSentenceDetector)

    assert(result == expectedResult)

  }

  it should "identify punctuation in a sentence" in {
    val sentence = "That is your answer?"
    val expectedResult = true

    val result = deepSentenceDetector.sentenceHasPunctuation(sentence)

    assert(result == expectedResult)
  }

  it should "fit a sentence detector model" in {
    val testDataSet = Seq("dummy").toDS.toDF("text")
    val pipeline = new RecursivePipeline().setStages(
      Array(documentAssembler,
        tokenizer,
        nerTagger,
        nerConverter,
        deepSentenceDetector
      ))

    val sentenceDetectorModel = pipeline.fit(testDataSet)
    val model = sentenceDetectorModel.stages.last.asInstanceOf[DeepSentenceDetector]

    assert(model.isInstanceOf[DeepSentenceDetector])
  }

  "A Deep Sentence Detector that receives a dataset of unpunctuated sentences" should behave like {

    val testDataSet = Seq("I am Batman I live in Gotham",
                          "i love deep learning winter is coming").toDS.toDF("text")

    val expectedResult = Seq(
      Seq("I am Batman", "I live in Gotham"),
      Seq("i love deep learning", "winter is coming")
    )

    transformDataSet(testDataSet, pipeline, expectedResult)
  }

  "A Deep Sentence Detector that receives a dataset of punctuated sentences" should behave like {

    val testDataSet = Seq("This is a sentence. This is another sentence").toDS.toDF("text")

    val expectedResult = Seq(
      Seq("This is a sentence.", "This is another sentence")
    )

    transformDataSet(testDataSet, pipeline, expectedResult)
  }

  "A Deep Sentence Detector that receives a dataset of punctuated and unpunctuated sentences" should behave like {

    val testDataSet = Seq("This is a sentence. This is another sentence",
                          "I love deep learning Winter is coming").toDS.toDF("text")

    val expectedResult = Seq(
      Seq("This is a sentence.", "This is another sentence"),
      Seq("I love deep learning", "Winter is coming")
    )

    transformDataSet(testDataSet, pipeline, expectedResult)
  }

  "A Deep Sentence Detector (trained with small epochs) that receives a dataset of unpunctuated sentences" should
    behave like {

    val testDataSet = Seq("I am Batman I live in Gotham",
      "i love deep learning winter is coming").toDS.toDF("text")

    val expectedResult = Seq(
      Seq("I am Batman", "I live in Gotham"),
      Seq("i love deep learning", "winter is coming")
    )

    transformDataSet(testDataSet, pipelineSmallEpochs, expectedResult)
  }

  "A Deep Sentence Detector (trained with small epochs) that receives a dataset of punctuated sentences" should
    behave like {

    val testDataSet = Seq("This is a sentence. This is another sentence").toDS.toDF("text")

    val expectedResult = Seq(
      Seq("This is a sentence.", "This is another sentence")
    )

    transformDataSet(testDataSet, pipelineSmallEpochs, expectedResult)
  }

  "A Deep Sentence Detector (trained with small epochs) that receives a dataset of punctuated and unpunctuated sentences" should
    behave like {

    val testDataSet = Seq("This is a sentence. This is another sentence",
      "I love deep learning Winter is coming").toDS.toDF("text")

    val expectedResult = Seq(
      Seq("This is a sentence.", "This is another sentence"),
      Seq("I love deep learning", "Winter is coming")
    )

    transformDataSet(testDataSet, pipelineSmallEpochs, expectedResult)
  }

  "A Deep Sentence Detector (trained with small epochs) that receives a dataset of punctuated and unpunctuated sentences in one row" should
    behave like {

    val testDataSet = Seq("This is a sentence. I love deep learning Winter is coming",
      "This is another sentence").toDS.toDF("text")
    val expectedResult = Seq(
      Seq("This is a sentence.", "I love deep learning", "Winter is coming"),
      Seq("This is another sentence")
    )

    transformDataSet(testDataSet, pipelineSmallEpochs, expectedResult)
  }

}
