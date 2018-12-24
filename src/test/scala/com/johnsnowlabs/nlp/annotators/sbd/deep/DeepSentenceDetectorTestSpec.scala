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

  "A Deep Sentence Detector" should "retrieve NER entities from annotations" in {

    val expectedEntities = Seq(Annotation(CHUNK, 0, 0, "I", Map("entity"->"sent")),
                               Annotation(CHUNK, 12, 12, "I", Map("entity"->"sent")))

    val entities = deepSentenceDetector.getNerEntities(annotations)

    assert(entities == expectedEntities)

  }

  it should "retrieve a sentence from annotations" in {

    val expectedSentence = "I am Batman I live in Gotham"

    val sentence = deepSentenceDetector.retrieveSentence(annotations)

    assert(sentence == expectedSentence)

  }

  it should "segment sentences" in {
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

//  "A Deep Sentence Detector (trained with small epochs) that receives a dataset of punctuated sentences" should
//    behave like {
//
//    val testDataSet = Seq("This is a sentence. This is another sentence").toDS.toDF("text")
//
//    val expectedResult = Seq(
//      Seq("This is a sentence.", "This is another sentence")
//    )
//
//    transformDataSet(testDataSet, pipelineSmallEpochs, expectedResult)
//  }

}
