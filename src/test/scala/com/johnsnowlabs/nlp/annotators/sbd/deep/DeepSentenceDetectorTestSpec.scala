package com.johnsnowlabs.nlp.annotators.sbd.deep

import com.johnsnowlabs.nlp.AnnotatorType.{CHUNK, DOCUMENT, TOKEN}
import com.johnsnowlabs.nlp.SparkAccessor.spark.implicits._
import com.johnsnowlabs.nlp.{Annotation, _}
import com.johnsnowlabs.nlp.annotator.{NerConverter, SentenceDetector, Tokenizer}
import com.johnsnowlabs.nlp.annotators.ner.dl.NerDLApproach
import com.johnsnowlabs.nlp.embeddings.WordEmbeddingsFormat
import org.apache.spark.sql.{Dataset, Row}
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

  private lazy val annotations = {
    val paragraph = "I am Batman I live in Gotham"
    val testDataSet = Seq(paragraph).toDS.toDF("text")
    val pipeline = new RecursivePipeline().setStages(Array(documentAssembler, tokenizer,
                                                           nerTaggerSmallEpochs, nerConverter))
    val documentAnnotation = Seq(Annotation(DOCUMENT, 0, paragraph.length-1, paragraph, Map()))
    val tokenAnnotations = getAnnotations(testDataSet, pipeline, "token")
    val nerConverterAnnotations = getAnnotations(testDataSet, pipeline, "ner_con")
    documentAnnotation++tokenAnnotations++nerConverterAnnotations
  }

  private def getAnnotations(dataset: Dataset[_], pipeline: RecursivePipeline, annotator: String) = {
    val annotations = pipeline.fit(dataset).transform(dataset).select(annotator).rdd.map(_.getSeq[Row](0))
    annotations.flatMap(_.map{
      case Row(annotatorType: String, begin: Int, end: Int, result: String, metadata: Map[String, String]) =>
        Annotation(annotatorType, begin, end, result, metadata)
    } ).collect()
  }

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
      Seq(Annotation(DOCUMENT, 0, expectedSentence1.length-1, expectedSentence1, Map("sentence"->"")),
          Annotation(DOCUMENT, 0, expectedSentence2.length-1, expectedSentence2, Map("sentence"->"")))

    val segmentedSentence = deepSentenceDetector.segmentSentence(entities, sentence)

    assert(segmentedSentence == expectedSegmentedSentences)

  }

  it should "identify punctuation in a sentence with punctuation" in {

    val sentence = "That is your answer?"
    val expectedResult = true

    val result = deepSentenceDetector.sentenceHasPunctuation(sentence)

    assert(result == expectedResult)
  }

  it should "not identify punctuation in a sentence without punctuation" in {
    val sentence = "I love deep learning Winter is coming;"
    val expectedResult = false

    val result = deepSentenceDetector.sentenceHasPunctuation(sentence)

    assert(result == expectedResult)
  }

  it should "get an unpuncutated sentence from a paragraph with one unpunctuated sentence" in {

    val paragraph = "This is a sentence. I love deep learning Winter is coming"
    val document = Seq(Annotation(DOCUMENT, 0, paragraph.length-1, paragraph, Map()))
    val pragmaticSentenceDetector = new SentenceDetector().annotate(document)
    val expectedUnpunctuatedSentences = Seq(Annotation(DOCUMENT, 20, 56, "I love deep learning Winter is coming", Map()))

    val unpunctuatedSentence = deepSentenceDetector.getUnpunctuatedSentences(pragmaticSentenceDetector)

    assert(unpunctuatedSentence == expectedUnpunctuatedSentences)

  }

  it should "get an unpuncutated sentences from a paragraph with several unpunctuated sentences" in {

    val paragraph = "This is a sentence. I love deep learning Winter is coming; I am Batman I live in Gotham"
    val document = Seq(Annotation(DOCUMENT, 0, paragraph.length-1, paragraph, Map()))
    val pragmaticSentenceDetector = new SentenceDetector().annotate(document)
    val expectedUnpunctuatedSentences = Seq(
      Annotation(DOCUMENT, 20, 57, "I love deep learning Winter is coming;", Map()),
      Annotation(DOCUMENT, 59, 86, "I am Batman I live in Gotham", Map())
      )

    val unpunctuatedSentences = deepSentenceDetector.getUnpunctuatedSentences(pragmaticSentenceDetector)

    assert(unpunctuatedSentences == expectedUnpunctuatedSentences)

  }

  it should "retrieve valid NER entities from a paragraph with one punctuated and one unpunctuated sentence" in {

    val paragraph = "This is a sentence. I love deep learning Winter is coming"
    val testDataSet = Seq(paragraph).toDS.toDF("text")
    val pipeline = new RecursivePipeline().setStages(Array(documentAssembler, tokenizer, nerTagger, nerConverter))
    val documentAnnotation = Seq(Annotation(DOCUMENT, 0, paragraph.length-1, paragraph, Map()))
    val tokenAnnotations = getAnnotations(testDataSet, pipeline, "token")
    val nerConverterAnnotations = getAnnotations(testDataSet, pipeline, "ner_con")
    val annotations  = documentAnnotation ++ tokenAnnotations ++ nerConverterAnnotations
    val pragmaticSentenceDetector = new SentenceDetector().annotate(documentAnnotation)
    val unpunctuatedSentence = deepSentenceDetector.getUnpunctuatedSentences(pragmaticSentenceDetector)
    val expectedResult = Seq(Seq(
      Annotation(CHUNK, 0, 0, "I", Map("entity"->"sent")),
      Annotation(CHUNK, 21, 26, "Winter", Map("entity"->"sent")))
    )

    val result = deepSentenceDetector.retrieveValidNerEntities(annotations, unpunctuatedSentence)

    assert(result == expectedResult)

  }

  it should "retrieve valid NER entities from a paragraph with one punctuated and several unpunctuated sentences" in {

    val paragraph = "This is a sentence. I love deep learning Winter is coming; I am Batman I live in Gotham"
    val testDataSet = Seq(paragraph).toDS.toDF("text")
    val pipeline = new RecursivePipeline().setStages(Array(documentAssembler, tokenizer, nerTagger, nerConverter))
    val documentAnnotation = Seq(Annotation(DOCUMENT, 0, paragraph.length-1, paragraph, Map()))
    val tokenAnnotations = getAnnotations(testDataSet, pipeline, "token")
    val nerConverterAnnotations = Seq(
      Annotation(CHUNK, 0, 3, "This", Map("entity"->"sent")),
      Annotation(CHUNK, 5, 6, "is", Map("entity"->"sent")),
      Annotation(CHUNK, 20, 20, "I", Map("entity"->"sent")),
      Annotation(CHUNK, 41, 46, "Winter", Map("entity"->"sent")),
      Annotation(CHUNK, 59, 59, "I", Map("entity"->"sent")),
      Annotation(CHUNK, 70, 70, "I", Map("entity"->"sent"))
    )
    val annotations  = documentAnnotation ++ tokenAnnotations ++ nerConverterAnnotations
    val pragmaticSentenceDetector = new SentenceDetector().annotate(documentAnnotation)
    val unpunctuatedSentences = deepSentenceDetector.getUnpunctuatedSentences(pragmaticSentenceDetector)
    val expectedValidNerEntities = Seq(
      Seq(Annotation(CHUNK, 0, 0, "I", Map("entity"->"sent")),
          Annotation(CHUNK, 21, 26, "Winter", Map("entity"->"sent"))),
      Seq(Annotation(CHUNK, 0, 0, "I", Map("entity"->"sent")),
          Annotation(CHUNK, 11, 11, "I", Map("entity"->"sent")))
    )

    val validNerEntities = deepSentenceDetector.retrieveValidNerEntities(annotations, unpunctuatedSentences)

    assert(validNerEntities == expectedValidNerEntities)
  }

  it should "detect sentences from a paragraph with one punctuated and several unpunctuated sentences  in one row" in {
    val paragraph = "This is a sentence. I love deep learning Winter is coming; I am Batman I live in Gotham"
    val testDataSet = Seq(paragraph).toDS.toDF("text")
    val pipeline = new RecursivePipeline().setStages(Array(documentAssembler, tokenizer, nerTagger, nerConverter))
    val documentAnnotation = Seq(Annotation(DOCUMENT, 0, paragraph.length-1, paragraph, Map()))
    val tokenAnnotations = getAnnotations(testDataSet, pipeline, "token")
    val nerConverterAnnotations = Seq(
      Annotation(CHUNK, 0, 3, "This", Map("entity"->"sent")),
      Annotation(CHUNK, 5, 6, "is", Map("entity"->"sent")),
      Annotation(CHUNK, 20, 20, "I", Map("entity"->"sent")),
      Annotation(CHUNK, 41, 46, "Winter", Map("entity"->"sent")),
      Annotation(CHUNK, 59, 59, "I", Map("entity"->"sent")),
      Annotation(CHUNK, 71, 71, "I", Map("entity"->"sent"))
    )
    val annotations  = documentAnnotation ++ tokenAnnotations ++ nerConverterAnnotations
    val pragmaticSegmentedSentences = new SentenceDetector().annotate(documentAnnotation)
    val unpunctuatedSentences = deepSentenceDetector.getUnpunctuatedSentences(pragmaticSegmentedSentences)
    val validNerEntities = deepSentenceDetector.retrieveValidNerEntities(annotations, unpunctuatedSentences)
    val sentence1 = "I love deep learning"
    val sentence2 = "Winter is coming;"
    val sentence3 = "I am Batman"
    val sentence4 = "I live in Gotham"
    val expectedDetectedSentences = Seq(
      Annotation(DOCUMENT, 0, sentence1.length-1, sentence1, Map("sentence"->"")),
      Annotation(DOCUMENT, 0, sentence2.length-1, sentence2, Map("sentence"->"")),
      Annotation(DOCUMENT, 0, sentence3.length-1, sentence3, Map("sentence"->"")),
      Annotation(DOCUMENT, 0, sentence4.length-1, sentence4, Map("sentence"->""))
    )

    val detectedSentences = deepSentenceDetector.deepSentenceDetector(unpunctuatedSentences, validNerEntities)

    assert(detectedSentences == expectedDetectedSentences)
  }

  it should
    "merge pragmatic and deep sentence detector outputs of a paragraph with punctuated and unpunctuated sentences" in {

    val paragraph = "This is a sentence. I love deep learning Winter is coming"
    val document = Seq(Annotation(DOCUMENT, 0 , paragraph.length-1, paragraph, Map()))
    val pragmaticSegmentedSentences = new SentenceDetector().annotate(document)
    val sentence1 = "This is a sentence."
    val sentence2 = "I love deep learning"
    val sentence3 = "Winter is coming"
    val expectedMergedSentences = Seq(
      Annotation(DOCUMENT, 0, sentence1.length-1, sentence1, Map("sentence"->"1")),
      Annotation(DOCUMENT, 0, sentence2.length-1, sentence2, Map("sentence"->"2")),
      Annotation(DOCUMENT, 0, sentence3.length-1, sentence3, Map("sentence"->"3"))
    )
    val deepSegmentedSentences = Seq(
      Annotation(DOCUMENT, 0 , 19, "I love deep learning", Map("sentence"->"")),
      Annotation(DOCUMENT, 0 , 15, "Winter is coming", Map("sentence"->""))
    )
    val mergedSegmentedSentences = deepSentenceDetector.mergeSentenceDetectors(pragmaticSegmentedSentences,
      deepSegmentedSentences)

    assert(mergedSegmentedSentences == expectedMergedSentences)

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

    val testDataSet = Seq("This is a sentence. This is another sentence.").toDS.toDF("text")

    val expectedResult = Seq(
      Seq("This is a sentence.", "This is another sentence.")
    )

    transformDataSet(testDataSet, pipeline, expectedResult)
  }

  "A Deep Sentence Detector that receives a dataset of punctuated and unpunctuated sentences" should behave like {

    val testDataSet = Seq("This is a sentence. This is another sentence.",
                          "I love deep learning Winter is coming").toDS.toDF("text")

    val expectedResult = Seq(
      Seq("This is a sentence.", "This is another sentence."),
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

    val testDataSet = Seq("This is a sentence. This is another sentence.").toDS.toDF("text")

    val expectedResult = Seq(
      Seq("This is a sentence.", "This is another sentence.")
    )

    transformDataSet(testDataSet, pipelineSmallEpochs, expectedResult)
  }

  "A Deep Sentence Detector (trained with small epochs) that receives a dataset of punctuated and unpunctuated sentences" should
    behave like {

    val testDataSet = Seq("This is a sentence. This is another sentence.",
      "I love deep learning Winter is coming").toDS.toDF("text")

    val expectedResult = Seq(
      Seq("This is a sentence.", "This is another sentence."),
      Seq("I love deep learning", "Winter is coming")
    )

    transformDataSet(testDataSet, pipelineSmallEpochs, expectedResult)
  }

  "A Deep Sentence Detector (trained with small epochs) that receives a dataset of punctuated and unpunctuated sentences in one row" should
    behave like {

    val testDataSet = Seq("This is a sentence. I love deep learning Winter is coming",
      "This is another sentence.").toDS.toDF("text")
    val expectedResult = Seq(
      Seq("This is a sentence.", "I love deep learning", "Winter is coming"),
      Seq("This is another sentence.")
    )

    transformDataSet(testDataSet, pipelineSmallEpochs, expectedResult)
  }

}
