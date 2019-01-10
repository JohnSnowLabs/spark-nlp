package com.johnsnowlabs.nlp.annotators.sbd.deep

import com.johnsnowlabs.nlp.AnnotatorType.{CHUNK, DOCUMENT}
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

  private val strongNerTagger = new NerDLApproach()
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

  private val weakNerTagger = new NerDLApproach()
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
    .setIncludePragmaticSegmenter(false)
    .setEndPunctuation(Array(".", "?"))

  private val pureDeepSentenceDetector = new DeepSentenceDetector()
    .setInputCols(Array("document", "token", "ner_con"))
    .setOutputCol("seg_sentence")
    .setIncludePragmaticSegmenter(true)
    .setEndPunctuation(Array(".", "?"))

  private val finisher = new Finisher()
    .setInputCols("seg_sentence")
    .setOutputCols("sentence")

  private val strongPipeline = new RecursivePipeline().setStages(
    Array(documentAssembler,
      tokenizer,
      strongNerTagger,
      nerConverter,
      deepSentenceDetector,
      finisher
    ))

  private val weakPipeline = new RecursivePipeline().setStages(
    Array(documentAssembler,
      tokenizer,
      weakNerTagger,
      nerConverter,
      pureDeepSentenceDetector,
      finisher
    ))

  private lazy val annotations = {
    val paragraph = "I am Batman I live in Gotham"
    val testDataSet = Seq(paragraph).toDS.toDF("text")
    val pipeline = new RecursivePipeline().setStages(Array(documentAssembler, tokenizer, weakNerTagger, nerConverter))
    val documentAnnotation = Seq(Annotation(DOCUMENT, 0, paragraph.length-1, paragraph, Map()))
    val tokenAnnotations = getAnnotations(testDataSet, pipeline, "token")
    val nerConverterAnnotations = getAnnotations(testDataSet, pipeline, "ner_con")
    documentAnnotation++tokenAnnotations++nerConverterAnnotations
  }

  def getAnnotationsWithTokens(paragraph: String): Seq[Annotation] = {
    val testDataSet = Seq(paragraph).toDS.toDF("text")
    val pipeline = new RecursivePipeline().setStages(Array(documentAssembler, tokenizer))
    val tokenAnnotations = getAnnotations(testDataSet, pipeline, "token")
    documentAnnotation ++ tokenAnnotations
  }

  private def getAnnotations(dataset: Dataset[_], pipeline: RecursivePipeline, column: String) = {
    val annotations = pipeline.fit(dataset).transform(dataset).select(column).rdd.map(_.getSeq[Row](0))
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

  var paragraph = ""
  private var documentAnnotation = Seq[Annotation]()
  private var pragmaticSegmentedSentences = Seq[Annotation]()
  private var unpunctuatedSentences = Seq[Annotation]()

  "A Deep Sentence Detector with none end of sentence punctuation" should "get an unpuncutated sentence" in {
    paragraph = "This is a sentence. This is another sentence."

    val deepSentenceDetector = new DeepSentenceDetector()
      .setInputCols(Array("document", "token", "ner_con"))
      .setOutputCol("seg_sentence")
      .setIncludePragmaticSegmenter(true)
      .setEndPunctuation(Array(""))
    documentAnnotation = Seq(Annotation(DOCUMENT, 0, paragraph.length-1, paragraph, Map()))
    pragmaticSegmentedSentences = new SentenceDetector().annotate(documentAnnotation)
    val expectedUnpunctuatedSentences = Seq(
      Annotation(DOCUMENT, 0, 18, "This is a sentence.", Map()),
      Annotation(DOCUMENT, 20, 44, "This is another sentence.", Map())
    )

    unpunctuatedSentences = deepSentenceDetector.getUnpunctuatedSentences(pragmaticSegmentedSentences)

    assert(unpunctuatedSentences == expectedUnpunctuatedSentences)

  }

  it should "retrieve empty valid NER entities when NER converter does not find entities" in {
    val deepSentenceDetector = new DeepSentenceDetector()
      .setInputCols(Array("document", "token", "ner_con"))
      .setOutputCol("seg_sentence")
      .setIncludePragmaticSegmenter(true)
      .setEndPunctuation(Array(""))
    val nerConverterAnnotations = Array[Annotation]()
    val annotations  = getAnnotationsWithTokens(paragraph) ++ nerConverterAnnotations

    validNerEntities = deepSentenceDetector.retrieveValidNerEntities(annotations, unpunctuatedSentences)

    assert(validNerEntities.isEmpty)
  }

  it should "retrieve valid NER entities when NER converter finds entities" in {
    val deepSentenceDetector = new DeepSentenceDetector()
      .setInputCols(Array("document", "token", "ner_con"))
      .setOutputCol("seg_sentence")
      .setIncludePragmaticSegmenter(true)
      .setEndPunctuation(Array(""))
    val nerConverterAnnotations = Seq(
      Annotation(CHUNK, 0, 3, "This", Map("entity"->"sent")),
      Annotation(CHUNK, 20, 23, "This", Map("entity"->"sent"))
    )
    val annotations  = getAnnotationsWithTokens(paragraph) ++ nerConverterAnnotations
    val expectedValidNetEntities = Seq(
      Seq(Annotation(CHUNK, 0, 3, "This", Map("entity"->"sent"))),
      Seq(Annotation(CHUNK, 0, 3, "This", Map("entity"->"sent")))
    )

    validNerEntities = deepSentenceDetector.retrieveValidNerEntities(annotations, unpunctuatedSentences)

    assert(validNerEntities == expectedValidNetEntities)
  }

  it should "fit a sentence detector model" in {

    val testDataSet = Seq("This is a sentence. This is another sentence.",
      "I love deep learning Winter is coming").toDS.toDF("text")
    val deepSentenceDetector = new DeepSentenceDetector()
      .setInputCols(Array("document", "token", "ner_con"))
      .setOutputCol("seg_sentence")
      .setIncludePragmaticSegmenter(true)
      .setEndPunctuation(Array(""))
    val weakPipeline = new RecursivePipeline().setStages(
      Array(documentAssembler,
        tokenizer,
        weakNerTagger,
        nerConverter,
        deepSentenceDetector
      ))

    val sentenceDetectorModel = weakPipeline.fit(testDataSet)
    val model = sentenceDetectorModel.stages.last.asInstanceOf[DeepSentenceDetector]

    assert(model.isInstanceOf[DeepSentenceDetector])
  }


  "A Deep Sentence Detector with a paragraph of one punctuated and one unpunctuated sentence" should
    "get an unpuncutated sentence" in {

    paragraph = "This is a sentence. I love deep learning Winter is coming"
    documentAnnotation = Seq(Annotation(DOCUMENT, 0, paragraph.length-1, paragraph, Map()))
    pragmaticSegmentedSentences = new SentenceDetector().annotate(documentAnnotation)
    val expectedUnpunctuatedSentences = Seq(Annotation(DOCUMENT, 20, 56, "I love deep learning Winter is coming", Map()))

    unpunctuatedSentences = deepSentenceDetector.getUnpunctuatedSentences(pragmaticSegmentedSentences)

    assert(unpunctuatedSentences == expectedUnpunctuatedSentences)

  }

  private var validNerEntities = Seq[Seq[Annotation]]()

  it should "retrieve valid NER entities" in {

    val nerConverterAnnotations = Seq(
      Annotation(CHUNK, 0, 3, "This", Map("entity"->"sent")),
      Annotation(CHUNK, 20, 20, "I", Map("entity"->"sent")),
      Annotation(CHUNK, 41, 46, "Winter", Map("entity"->"sent"))
    )
    val annotations  = getAnnotationsWithTokens(paragraph) ++ nerConverterAnnotations
    val expectedValidNerEntities = Seq(Seq(
      Annotation(CHUNK, 0, 0, "I", Map("entity"->"sent")),
      Annotation(CHUNK, 21, 26, "Winter", Map("entity"->"sent")))
    )

    validNerEntities = deepSentenceDetector.retrieveValidNerEntities(annotations, unpunctuatedSentences)

    assert(validNerEntities == expectedValidNerEntities)

  }

  it should
    "detect sentences" in {

    val sentence1 = "I love deep learning"
    val sentence2 = "Winter is coming"
    val expectedDetectedSentences = Seq(
      Annotation(DOCUMENT, 0, sentence1.length-1, sentence1, Map("sentence"->"")),
      Annotation(DOCUMENT, 0, sentence2.length-1, sentence2, Map("sentence"->""))
    )

    val detectedSentences = deepSentenceDetector.deepSentenceDetector(unpunctuatedSentences, validNerEntities)

    assert(detectedSentences == expectedDetectedSentences)
  }

  it should
    "merge pragmatic and deep sentence detector outputs" in {

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

  "A Deep Sentence Detector with a paragraph of one sentence and several unpunctuated sentences" should
    "get unpuncutated sentences" in {

    paragraph = "This is a sentence. I love deep learning Winter is coming; I am Batman I live in Gotham"
    documentAnnotation = Seq(Annotation(DOCUMENT, 0, paragraph.length-1, paragraph, Map()))
    pragmaticSegmentedSentences = new SentenceDetector().annotate(documentAnnotation)
    val expectedUnpunctuatedSentences = Seq(
      Annotation(DOCUMENT, 20, 57, "I love deep learning Winter is coming;", Map()),
      Annotation(DOCUMENT, 59, 86, "I am Batman I live in Gotham", Map())
      )

    unpunctuatedSentences = deepSentenceDetector.getUnpunctuatedSentences(pragmaticSegmentedSentences)

    assert(unpunctuatedSentences == expectedUnpunctuatedSentences)

  }

  it should "retrieve valid NER entities" in {

    val nerConverterAnnotations = Seq(
      Annotation(CHUNK, 0, 3, "This", Map("entity"->"sent")),
      Annotation(CHUNK, 5, 6, "is", Map("entity"->"sent")),
      Annotation(CHUNK, 20, 20, "I", Map("entity"->"sent")),
      Annotation(CHUNK, 41, 46, "Winter", Map("entity"->"sent")),
      Annotation(CHUNK, 59, 59, "I", Map("entity"->"sent")),
      Annotation(CHUNK, 71, 71, "I", Map("entity"->"sent"))
    )
    val annotations  = getAnnotationsWithTokens(paragraph) ++ nerConverterAnnotations
    val expectedValidNerEntities = Seq(
      Seq(Annotation(CHUNK, 0, 0, "I", Map("entity"->"sent")),
          Annotation(CHUNK, 21, 26, "Winter", Map("entity"->"sent"))),
      Seq(Annotation(CHUNK, 0, 0, "I", Map("entity"->"sent")),
          Annotation(CHUNK, 12, 12, "I", Map("entity"->"sent")))
    )

    validNerEntities = deepSentenceDetector.retrieveValidNerEntities(annotations, unpunctuatedSentences)

    assert(validNerEntities == expectedValidNerEntities)
  }

  it should
    "detect sentences" in {

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
    "merge pragmatic and deep sentence detector outputs" in {

    val deepSegmentedSentences = deepSentenceDetector.deepSentenceDetector(unpunctuatedSentences, validNerEntities)
    val sentence1= "This is a sentence."
    val sentence2 = "I love deep learning"
    val sentence3 = "Winter is coming;"
    val sentence4 = "I am Batman"
    val sentence5 = "I live in Gotham"
    val expectedMergedSentences = Seq(
      Annotation(DOCUMENT, 0, sentence1.length-1, sentence1, Map("sentence"->"1")),
      Annotation(DOCUMENT, 0, sentence2.length-1, sentence2, Map("sentence"->"2")),
      Annotation(DOCUMENT, 0, sentence3.length-1, sentence3, Map("sentence"->"3")),
      Annotation(DOCUMENT, 0, sentence4.length-1, sentence4, Map("sentence"->"4")),
      Annotation(DOCUMENT, 0, sentence5.length-1, sentence5, Map("sentence"->"5"))
    )

    val mergedSegmentedSentences = deepSentenceDetector.mergeSentenceDetectors(pragmaticSegmentedSentences,
      deepSegmentedSentences)

    assert(mergedSegmentedSentences == expectedMergedSentences)

  }

  it should "fit a sentence detector model" in {
    val testDataSet = Seq("dummy").toDS.toDF("text")
    val strongPipeline = new RecursivePipeline().setStages(
      Array(documentAssembler,
        tokenizer,
        strongNerTagger,
        nerConverter,
        deepSentenceDetector
      ))

    val sentenceDetectorModel = strongPipeline.fit(testDataSet)
    val model = sentenceDetectorModel.stages.last.asInstanceOf[DeepSentenceDetector]

    assert(model.isInstanceOf[DeepSentenceDetector])
  }

  "A Deep Sentence Detector (trained with strong NER) that receives a dataset of unpunctuated sentences" should behave like {

    val testDataSet = Seq("I am Batman I live in Gotham",
                          "i love deep learning winter is coming").toDS.toDF("text")

    val expectedResult = Seq(
      Seq("I am Batman", "I live in Gotham"),
      Seq("i love deep learning", "winter is coming")
    )

    transformDataSet(testDataSet, strongPipeline, expectedResult)
  }

  "A Deep Sentence Detector (trained with strong NER) that receives a dataset of punctuated sentences" should behave like {

    val testDataSet = Seq("This is a sentence. This is another sentence.").toDS.toDF("text")

    val expectedResult = Seq(
      Seq("This is a sentence.", "This is another sentence.")
    )

    transformDataSet(testDataSet, strongPipeline, expectedResult)
  }

  "A Deep Sentence Detector (trained with weak NER) that receives a dataset of unpunctuated sentences" should
    behave like {

    val testDataSet = Seq("I am Batman I live in Gotham",
      "i love deep learning winter is coming").toDS.toDF("text")

    val expectedResult = Seq(
      Seq("I am Batman", "I live in Gotham"),
      Seq("i love deep learning", "winter is coming")
    )

    transformDataSet(testDataSet, weakPipeline, expectedResult)
  }

  "A Deep Sentence Detector (trained with weak NER) that receives a dataset of punctuated sentences" should
    behave like {

    val testDataSet = Seq("This is a sentence. This is another sentence.").toDS.toDF("text")

    val expectedResult = Seq(
      Seq("This is a sentence.", "This is another sentence.")
    )

    transformDataSet(testDataSet, weakPipeline, expectedResult)
  }

  "A Deep Sentence Detector (trained with weak NER) that receives a dataset of punctuated and unpunctuated sentences" should
    behave like {

    val testDataSet = Seq("This is a sentence. This is another sentence.",
      "I love deep learning Winter is coming").toDS.toDF("text")

    val expectedResult = Seq(
      Seq("This is a sentence.", "This is another sentence."),
      Seq("I love deep learning", "Winter is coming")
    )

    transformDataSet(testDataSet, weakPipeline, expectedResult)
  }

  "A Deep Sentence Detector (trained with weak NER) that receives a dataset of punctuated and unpunctuated sentences in one row" should
    behave like {

    val testDataSet = Seq("This is a sentence. I love deep learning Winter is coming",
      "This is another sentence.").toDS.toDF("text")
    val expectedResult = Seq(
      Seq("This is a sentence.", "I love deep learning", "Winter is coming"),
      Seq("This is another sentence.")
    )

    transformDataSet(testDataSet, weakPipeline, expectedResult)
  }

  "A Deep Sentence Detector (trained with a strong NER) that sets end of sentence punctuation as null" should behave like {

    val testDataSet = Seq("This is a sentence. This is another sentence.",
      "I love deep learning Winter is coming").toDS.toDF("text")
    val deepSentenceDetector = new DeepSentenceDetector()
      .setInputCols(Array("document", "token", "ner_con"))
      .setOutputCol("seg_sentence")
      .setIncludePragmaticSegmenter(true)
      .setEndPunctuation(Array(""))
    val pipeline = new RecursivePipeline().setStages(
      Array(documentAssembler,
        tokenizer,
        strongNerTagger,
        nerConverter,
        deepSentenceDetector,
        finisher
      ))
    val expectedResult = Seq(
      Seq("This is a sentence.", "This is another sentence."),
      Seq("I love deep learning", "Winter is coming")
    )

    transformDataSet(testDataSet, pipeline, expectedResult)

  }

  "A Deep Sentence Detector (trained with strong NER) that receives a dataset of punctuated and unpunctuated sentences" should
    behave like {

    val testDataSet = Seq("This is a sentence. This is another sentence.",
      "I love deep learning Winter is coming").toDS.toDF("text")

    val expectedResult = Seq(
      Seq("This is a sentence.", "This is another sentence."),
      Seq("I love deep learning", "Winter is coming")
    )

    transformDataSet(testDataSet, strongPipeline, expectedResult)
  }

}
