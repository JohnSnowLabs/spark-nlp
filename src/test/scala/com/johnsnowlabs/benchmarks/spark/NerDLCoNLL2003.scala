package com.johnsnowlabs.benchmarks.spark

import com.johnsnowlabs.nlp._
import com.johnsnowlabs.nlp.annotators.common.NerTagged
import com.johnsnowlabs.nlp.annotators.ner.dl.{NerDLApproach, NerDLModel}
import com.johnsnowlabs.nlp.annotators.ner.{NerConverter, Verbose}
import com.johnsnowlabs.nlp.datasets.CoNLL
import com.johnsnowlabs.nlp.embeddings.{WordEmbeddingsFormat, WordEmbeddings}
import com.johnsnowlabs.nlp.util.io.{ExternalResource, ReadAs}
import org.apache.spark.ml.PipelineModel


object NerDLPipeline extends App {
  val folder = "./"

  val trainFile = ExternalResource(folder + "eng.train", ReadAs.LINE_BY_LINE, Map.empty[String, String])
  val testFileA = ExternalResource(folder + "eng.testa", ReadAs.LINE_BY_LINE, Map.empty[String, String])
  val testFileB = ExternalResource(folder + "eng.testb", ReadAs.LINE_BY_LINE, Map.empty[String, String])

  val nerReader = CoNLL()

  def createPipeline() = {

    val glove = new WordEmbeddings()
      .setEmbeddingsSource("glove.6B.100d.txt", 100, WordEmbeddingsFormat.TEXT)
      .setInputCols("sentence", "token")
      .setOutputCol("glove")

    val nerTagger = new NerDLApproach()
      .setInputCols("sentence", "token", "glove")
      .setLabelColumn("label")
      .setMaxEpochs(1)
      .setRandomSeed(0)
      .setPo(0.005f)
      .setLr(1e-3f)
      .setDropout(0.5f)
      .setBatchSize(32)
      .setOutputCol("ner")
      .setVerbose(Verbose.Epochs)

    val converter = new NerConverter()
      .setInputCols("document", "token", "ner")
      .setOutputCol("ner_span")

    val labelConverter = new NerConverter()
      .setInputCols("document", "token", "label")
      .setOutputCol("label_span")

    Array(
      glove,
      nerTagger,
      converter,
      labelConverter
    )
  }

  def trainNerModel(er: ExternalResource): PipelineModel = {
    System.out.println("Dataset Reading")
    val time = System.nanoTime()
    val dataset = nerReader.readDataset(er, SparkAccessor.benchmarkSpark)
    System.out.println(s"Done, ${(System.nanoTime() - time)/1e9}\n")

    System.out.println("Start fitting")

    val stages = createPipeline()

    val pipeline = new RecursivePipeline()
      .setStages(stages)

    pipeline.fit(dataset)
  }

  def getUserFriendly(model: PipelineModel, file: ExternalResource): Array[Array[Annotation]] = {
    val df = model.transform(nerReader.readDataset(file, SparkAccessor.benchmarkSpark))
    Annotation.collect(df, "ner_span")
  }

  def measure(model: PipelineModel, file: ExternalResource, extended: Boolean = true, errorsToPrint: Int = 0): Unit = {
    val ner = model.stages.filter(s => s.isInstanceOf[NerDLModel]).head.asInstanceOf[NerDLModel].getModelIfNotSet
    val df = nerReader.readDataset(file, SparkAccessor.benchmarkSpark).toDF()
    val transformed = model.transform(df)

    val labeled = NerTagged.collectTrainingInstances(transformed, Seq("sentence", "token", "glove"), "label")

    ner.measure(labeled, (s: String) => System.out.println(s), extended, errorsToPrint)
  }

  val spark = SparkAccessor.benchmarkSpark

  val model = trainNerModel(trainFile)

  measure(model, trainFile, false)
  measure(model, testFileA, false)
  measure(model, testFileB, true)

  val annotations = getUserFriendly(model, testFileB)
  NerHelper.saveNerSpanTags(annotations, "predicted.csv")

  model.write.overwrite().save("ner_model")
  PipelineModel.read.load("ner_model")

  System.out.println("Training dataset")
  NerHelper.measureExact(nerReader, model, trainFile)

  System.out.println("Validation dataset")
  NerHelper.measureExact(nerReader, model, testFileA)

  System.out.println("Test dataset")
  NerHelper.measureExact(nerReader, model, testFileB)
}
