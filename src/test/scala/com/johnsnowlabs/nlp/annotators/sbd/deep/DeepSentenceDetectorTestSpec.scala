package com.johnsnowlabs.nlp.annotators.sbd.deep

import com.johnsnowlabs.nlp.AnnotatorType.{CHUNK, DOCUMENT, TOKEN}
import com.johnsnowlabs.nlp.SparkAccessor.spark.implicits._
import com.johnsnowlabs.nlp.{Annotation, _}
import com.johnsnowlabs.nlp.annotator.{NerConverter, SentenceDetector, Tokenizer}
import com.johnsnowlabs.nlp.annotators.ner.dl.NerDLApproach
import com.johnsnowlabs.nlp.embeddings.WordEmbeddingsFormat
import com.johnsnowlabs.util.PipelineModels
import org.apache.spark.sql.{Dataset, Row}
import org.scalatest.FlatSpec
import java.nio.file.{Files, Paths}


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
    .setIncludePragmaticSegmenter(true)
    .setEndPunctuation(Array(".", "?"))

  private val pureDeepSentenceDetector = new DeepSentenceDetector()
    .setInputCols(Array("document", "token", "ner_con"))
    .setOutputCol("seg_sentence")
    .setIncludePragmaticSegmenter(false)
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
      deepSentenceDetector,
      finisher
    ))

  def getAnnotationsWithNerConverter(paragraph: String, nerTagger: NerDLApproach): Seq[Annotation] = {
    //TODO: Consider hardcoded or a better way to mock nerConverterAnnotations result since is uncertain due tu NER
    val testDataSet = Seq(paragraph).toDS.toDF("text")
    val pipeline = new RecursivePipeline().setStages(Array(documentAssembler, tokenizer, nerTagger, nerConverter))
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

  private def getNerTagger(trainingFile: String) : NerDLApproach ={
    val nerTagger = new NerDLApproach()
      .setInputCols("document", "token")
      .setLabelColumn("label")
      .setOutputCol("ner")
      .setMaxEpochs(100)
      .setPo(0.01f)
      .setLr(0.1f)
      .setBatchSize(9)
      .setEmbeddingsSource("src/test/resources/ner-corpus/embeddings.100d.test.txt",
        100, WordEmbeddingsFormat.TEXT)
      .setExternalDataset(trainingFile)
      .setRandomSeed(0)
    nerTagger
  }

  "An empty document" should "raise exception" in {
    val annotations = Seq.empty[Annotation]
    val expectedErrorMessage = "Document is empty, please check."

    val caught = intercept[Exception]{
      deepSentenceDetector.getDocument(annotations)
    }

    assert(caught.getMessage == expectedErrorMessage)
  }

  "A pure Deep Sentence Detector with a right training file" should "retrieve NER entities from annotations" in {

    val nerTagger = getNerTagger("src/test/resources/ner-corpus/sentence-detector/hello_training_right.txt")
    val expectedEntities = Seq(Annotation(CHUNK, begin = 0, end = 4, "Hello", Map("entity"->"sent")),
      Annotation(CHUNK, begin = 32, end = 35, "This", Map("entity"->"sent")))
    val paragraph = "Hello world this is a sentence. This is another one."
    val annotations = getAnnotationsWithNerConverter(paragraph, nerTagger)

    val entities = deepSentenceDetector.getNerEntities(annotations)

    assert(entities == expectedEntities)

  }

  it should "retrieve a sentence from annotations" in {

    val nerTagger =getNerTagger("src/test/resources/ner-corpus/sentence-detector/hello_training_right.txt")
    val expectedSentence = "Hello world this is a sentence. This is another one."
    val annotations = getAnnotationsWithNerConverter(expectedSentence, nerTagger)

    val sentence = deepSentenceDetector.retrieveSentence(annotations)

    assert(sentence == expectedSentence)

  }

  it should "segment sentences" in {

    val nerTagger =getNerTagger("src/test/resources/ner-corpus/sentence-detector/hello_training_right.txt")
    val paragraph = "Hello world this is a sentence. This is another one."
    val annotations = getAnnotationsWithNerConverter(paragraph, nerTagger)
    val entities = deepSentenceDetector.getNerEntities(annotations)
    val sentence = deepSentenceDetector.retrieveSentence(annotations)
    val expectedSentence1 = "Hello world this is a sentence."
    val expectedSentence2 = "This is another one."
    val expectedSegmentedSentences =
      Seq(Annotation(DOCUMENT, 0, expectedSentence1.length-1, expectedSentence1, Map("sentence"->"1")),
        Annotation(DOCUMENT, expectedSentence1.length+1, sentence.length-1, expectedSentence2, Map("sentence"->"2")))

    val segmentedSentence = deepSentenceDetector.segmentSentence(entities, sentence)

    assert(segmentedSentence == expectedSegmentedSentences)

  }

  it should behave like {

    val nerTagger = getNerTagger("src/test/resources/ner-corpus/sentence-detector/hello_training_right.txt")
    val purePipeline = new RecursivePipeline().setStages(
        Array(documentAssembler,
          tokenizer,
          nerTagger,
          nerConverter,
          pureDeepSentenceDetector,
          finisher
        ))

    val testDataSet = Seq("Hello world this is a sentence. This is another one.").toDS.toDF("text")

    val expectedResult = Seq(
        Seq("Hello world this is a sentence.", "This is another one.")
      )

    transformDataSet(testDataSet, purePipeline, expectedResult)
  }

  "A pure Deep Sentence Detector with a half right training file" should "retrieve NER entities from annotations" in {

    val nerTagger = getNerTagger("src/test/resources/ner-corpus/sentence-detector/hello_training_half_right.txt")
    val expectedEntities = Seq(Annotation(TOKEN, begin = 0, end = 4, "Hello", Map("sentence"->"1")),
      Annotation(CHUNK, begin = 32, end = 35, "This", Map("entity"->"sent")))
    val paragraph = "Hello world this is a sentence. This is another one."
    val annotations = getAnnotationsWithNerConverter(paragraph, nerTagger)

    val entities = deepSentenceDetector.getNerEntities(annotations)

    assert(entities == expectedEntities)

  }

  it should "retrieve a sentence from annotations" in {

    val nerTagger =getNerTagger("src/test/resources/ner-corpus/sentence-detector/hello_training_half_right.txt")
    val expectedSentence = "Hello world this is a sentence. This is another one."
    val annotations = getAnnotationsWithNerConverter(expectedSentence, nerTagger)

    val sentence = deepSentenceDetector.retrieveSentence(annotations)

    assert(sentence == expectedSentence)

  }

  it should "segment sentences" in {

    val nerTagger = getNerTagger("src/test/resources/ner-corpus/sentence-detector/hello_training_half_right.txt")
    val paragraph = "Hello world this is a sentence. This is another one."
    val annotations = getAnnotationsWithNerConverter(paragraph, nerTagger)
    val entities = deepSentenceDetector.getNerEntities(annotations)
    val sentence = deepSentenceDetector.retrieveSentence(annotations)
    val expectedSentence1 = "Hello world this is a sentence."
    val expectedSentence2 = "This is another one."
    val expectedSegmentedSentences =
      Seq(Annotation(DOCUMENT, 0, expectedSentence1.length-1, expectedSentence1, Map("sentence"->"1")),
        Annotation(DOCUMENT, expectedSentence1.length+1, sentence.length-1, expectedSentence2, Map("sentence"->"2")))

    val segmentedSentence = deepSentenceDetector.segmentSentence(entities, sentence)

    assert(segmentedSentence == expectedSegmentedSentences)

  }

  it should behave like {

    val nerTagger = getNerTagger("src/test/resources/ner-corpus/sentence-detector/hello_training_half_right.txt")
    val purePipeline = new RecursivePipeline().setStages(
      Array(documentAssembler,
        tokenizer,
        nerTagger,
        nerConverter,
        pureDeepSentenceDetector,
        finisher
      ))

    val testDataSet = Seq("Hello world this is a sentence. This is another one.").toDS.toDF("text")

    val expectedResult = Seq(
      Seq("Hello world this is a sentence.", "This is another one.")
    )

    transformDataSet(testDataSet, purePipeline, expectedResult)
  }

  "A pure Deep Sentence Detector with a bad training file" should "retrieve NER entities from annotations" in {

    val paragraph = "Hello world this is a sentence. This is another one."
    val nerTagger = getNerTagger("src/test/resources/ner-corpus/sentence-detector/hello_training_bad.txt")
    val expectedEntities = Seq(Annotation(DOCUMENT, 0, paragraph.length-1,paragraph, Map.empty))
    val annotations = getAnnotationsWithNerConverter(paragraph, nerTagger)

    val entities = deepSentenceDetector.getNerEntities(annotations)

    assert(entities == expectedEntities)

  }

  it should "retrieve a sentence from annotations" in {

    val nerTagger =getNerTagger("src/test/resources/ner-corpus/sentence-detector/hello_training_bad.txt")
    val expectedSentence = "Hello world this is a sentence. This is another one."
    val annotations = getAnnotationsWithNerConverter(expectedSentence, nerTagger)

    val sentence = deepSentenceDetector.retrieveSentence(annotations)

    assert(sentence == expectedSentence)

  }

  it should "segment sentences" in {

    val nerTagger =getNerTagger("src/test/resources/ner-corpus/sentence-detector/hello_training_bad.txt")
    val paragraph = "Hello world this is a sentence. This is another one."
    val annotations = getAnnotationsWithNerConverter(paragraph, nerTagger)
    val entities = deepSentenceDetector.getNerEntities(annotations)
    val sentence = deepSentenceDetector.retrieveSentence(annotations)
    val expectedSentence = "Hello world this is a sentence. This is another one."
    val expectedSegmentedSentences =
      Seq(Annotation(DOCUMENT, 0, expectedSentence.length-1, expectedSentence, Map("sentence"->"1")))

    val segmentedSentence = deepSentenceDetector.segmentSentence(entities, sentence)

    assert(segmentedSentence == expectedSegmentedSentences)

  }

  "A Deep Sentence Detector" should "retrieve NER entities from annotations" in {

    val expectedEntities = Seq(Annotation(CHUNK, 0, 0, "I", Map("entity"->"sent")),
                               Annotation(CHUNK, 12, 12, "I", Map("entity"->"sent")))
    val paragraph = "I am Batman I live in Gotham"
    val annotations = getAnnotationsWithNerConverter(paragraph, weakNerTagger)

    val entities = deepSentenceDetector.getNerEntities(annotations)

    assert(entities == expectedEntities)

  }

  it should "retrieve a sentence from annotations" in {

    val expectedSentence = "I am Batman I live in Gotham"
    val paragraph = "I am Batman I live in Gotham"
    val annotations = getAnnotationsWithNerConverter(paragraph, weakNerTagger)

    val sentence = deepSentenceDetector.retrieveSentence(annotations)

    assert(sentence == expectedSentence)

  }

  it should "identify punctuation in a sentence with punctuation" in  {

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
      Annotation(DOCUMENT, begin = 0, end = 18, "This is a sentence.", Map()),
      Annotation(DOCUMENT, begin = 20, end = 44, "This is another sentence.", Map())
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
      Annotation(CHUNK, begin = 0, end = 3, "This", Map("entity"->"sent")),
      Annotation(CHUNK, begin = 20, end = 23, "This", Map("entity"->"sent"))
    )
    val annotations  = getAnnotationsWithTokens(paragraph) ++ nerConverterAnnotations
    val expectedValidNetEntities = Seq(
      Seq(Annotation(CHUNK, begin = 0, end = 3, "This", Map("entity"->"sent"))),
      Seq(Annotation(CHUNK, begin = 20, end = 23, "This", Map("entity"->"sent")))
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
      Annotation(CHUNK, begin = 0, end = 3, "This", Map("entity"->"sent")),
      Annotation(CHUNK, begin = 20, end = 20, "I", Map("entity"->"sent")),
      Annotation(CHUNK, begin = 41, end = 46, "Winter", Map("entity"->"sent"))
    )
    val annotations  = getAnnotationsWithTokens(paragraph) ++ nerConverterAnnotations
    val expectedValidNerEntities = Seq(Seq(
      Annotation(CHUNK, begin = 20, end = 20, "I", Map("entity"->"sent")),
      Annotation(CHUNK, begin = 41, end = 46, "Winter", Map("entity"->"sent")))
    )

    validNerEntities = deepSentenceDetector.retrieveValidNerEntities(annotations, unpunctuatedSentences)

    assert(validNerEntities == expectedValidNerEntities)

  }

  it should "detect sentences" in {
    val sentence1 = "I love deep learning"
    val sentence2 = "Winter is coming"
    val offset = 20
    val expectedDetectedSentences = Seq(
      Annotation(DOCUMENT, offset, sentence1.length-1 + offset, sentence1, Map.empty),
      Annotation(DOCUMENT, sentence1.length+1 + offset, paragraph.length-1, sentence2, Map.empty)
    )

    val detectedSentences = deepSentenceDetector.deepSentenceDetector(paragraph, unpunctuatedSentences, validNerEntities)

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
      Annotation(DOCUMENT, begin = 0 , end = 19, "I love deep learning", Map("sentence"->"1")),
      Annotation(DOCUMENT, begin = 0 , end = 15, "Winter is coming", Map("sentence"->"2"))
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
      Annotation(DOCUMENT, begin = 20, end = 57, "I love deep learning Winter is coming;", Map()),
      Annotation(DOCUMENT, begin = 59, end = 86, "I am Batman I live in Gotham", Map())
      )

    unpunctuatedSentences = deepSentenceDetector.getUnpunctuatedSentences(pragmaticSegmentedSentences)

    assert(unpunctuatedSentences == expectedUnpunctuatedSentences)

  }

  it should "retrieve valid NER entities" in {

    val nerConverterAnnotations = Seq(
      Annotation(CHUNK, begin = 0, end = 3, "This", Map("entity"->"sent")),
      Annotation(CHUNK, begin = 5, end = 6, "is", Map("entity"->"sent")),
      Annotation(CHUNK, begin = 20, end = 20, "I", Map("entity"->"sent")),
      Annotation(CHUNK, begin = 41, end = 46, "Winter", Map("entity"->"sent")),
      Annotation(CHUNK, begin = 59, end = 59, "I", Map("entity"->"sent")),
      Annotation(CHUNK, begin = 71, end = 71, "I", Map("entity"->"sent"))
    )
    val annotations  = getAnnotationsWithTokens(paragraph) ++ nerConverterAnnotations
    val expectedValidNerEntities = Seq(
      Seq(Annotation(CHUNK, begin = 20, end = 20, "I", Map("entity"->"sent")),
          Annotation(CHUNK, begin = 41, end = 46, "Winter", Map("entity"->"sent"))),
      Seq(Annotation(CHUNK, begin = 59, end = 59, "I", Map("entity"->"sent")),
          Annotation(CHUNK, begin = 71, end = 71, "I", Map("entity"->"sent")))
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
      Annotation(DOCUMENT, begin = 20, end = 39, sentence1, Map.empty),
      Annotation(DOCUMENT, begin = 41, end = 57, sentence2, Map.empty),
      Annotation(DOCUMENT, begin = 59, end = 69, sentence3, Map.empty),
      Annotation(DOCUMENT, begin = 71, end = 86, sentence4, Map.empty)
    )

    val detectedSentences = deepSentenceDetector.deepSentenceDetector(paragraph,
      unpunctuatedSentences, validNerEntities)

    assert(detectedSentences == expectedDetectedSentences)
  }

  it should
    "merge pragmatic and deep sentence detector outputs" in {

    val deepSegmentedSentences = deepSentenceDetector.deepSentenceDetector(paragraph,
      unpunctuatedSentences, validNerEntities)
    val sentence1= "This is a sentence."
    val sentence2 = "I love deep learning"
    val sentence3 = "Winter is coming;"
    val sentence4 = "I am Batman"
    val sentence5 = "I live in Gotham"
    val expectedMergedSentences = Seq(
      Annotation(DOCUMENT, begin = 0, end = 18, sentence1, Map("sentence"->"1")),
      Annotation(DOCUMENT, begin = 20, end = 39, sentence2, Map("sentence"->"2")),
      Annotation(DOCUMENT, begin = 41, end = 57, sentence3, Map("sentence"->"3")),
      Annotation(DOCUMENT, begin = 59, end = 69, sentence4, Map("sentence"->"4")),
      Annotation(DOCUMENT, begin = 71, end = 86, sentence5, Map("sentence"->"5"))
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

  "A Deep Sentence Detector (trained with strong NER) that receives a dataset of unpunctuated sentences" should
    behave like {

    val testDataSet = Seq("I am Batman I live in Gotham",
                          "i love deep learning winter is coming").toDS.toDF("text")

    val expectedResult = Seq(
      Seq("I am Batman", "I live in Gotham"),
      Seq("i love deep learning", "winter is coming")
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

  "A Deep Sentence Detector" should "be serializable" in {
    val emptyDataset = PipelineModels.dummyDataset
    val pipeline = new RecursivePipeline().setStages(
      Array(documentAssembler,
        tokenizer,
        strongNerTagger,
        nerConverter,
        deepSentenceDetector
      )).fit(emptyDataset)

    val deepSentenceDetectorModel = pipeline.stages.last.asInstanceOf[DeepSentenceDetector]

    deepSentenceDetectorModel.write.overwrite().save("./tmp_deep_sd_model")
    assertResult(true) {
      Files.exists(Paths.get("./tmp_deep_sd_model"))

    }
  }

  it should "be loaded from disk" in {
    val deepSentenceDetectorModel = DeepSentenceDetector.read.load("./tmp_deep_sd_model")
    assert(deepSentenceDetectorModel.isInstanceOf[DeepSentenceDetector])
  }

}
