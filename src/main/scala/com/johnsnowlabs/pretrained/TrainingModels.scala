package com.johnsnowlabs.pretrained

import java.io.File
import java.nio.file.Paths
import java.sql.Timestamp
import java.util.Date

import com.johnsnowlabs.nlp.DocumentAssembler
import com.johnsnowlabs.nlp.annotators.Tokenizer
import com.johnsnowlabs.nlp.annotators.ner.crf.{NerCrfApproach, NerCrfModel}
import com.johnsnowlabs.nlp.annotators.pos.perceptron.{PerceptronApproach, PerceptronModel}
import com.johnsnowlabs.nlp.annotators.sbd.pragmatic.SentenceDetector
import com.johnsnowlabs.nlp.embeddings.WordEmbeddingsFormat
import com.johnsnowlabs.util.{PipelineModels, Version, ZipArchiveUtil}
import org.apache.commons.io.FileUtils
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.util.{DefaultParamsWritable, MLWriter}
import org.apache.spark.sql.SparkSession


class EnModelsTraining(spark: SparkSession) {

  val language = Some("en")
  lazy val emptyDataset = PipelineModels.dummyDataset
  // Train models for all spark-nlp versions 1.*
  lazy val libVersion = Some(Version(1))
  lazy val sparkVersion = Some(Version.parse(spark.version).take(1))

  def trainAllAndSave(folder: String, clear: Boolean = true): Unit = {
    if (!new File(folder).exists) {
      FileUtils.forceMkdir(new File(folder))
    }

    require(new File(folder).isDirectory, s"folder $folder should exists and be directory")

    if (clear) {
      FileUtils.cleanDirectory(new File(folder))
    }

    TrainingHelper.saveModel("document_std", language, libVersion, sparkVersion, documentAssembler.write, folder)
    TrainingHelper.saveModel("sentence_std", language, libVersion, sparkVersion, stdSentenceDetector.write, folder)
    TrainingHelper.saveModel("tokenizer_std", language, libVersion, sparkVersion, stdTokenizer.write, folder)
    TrainingHelper.saveModel("pos_fast", language, libVersion, sparkVersion, fastPos.write, folder)
    TrainingHelper.saveModel("ner_fast", language, libVersion, sparkVersion, fastNer.write, folder)
  }

  lazy val documentAssembler = getDocumentAssembler()
  lazy val stdSentenceDetector = trainStdSentenceDetector()
  lazy val stdTokenizer = trainStdTokenizer()
  lazy val fastPos = trainFastPos()
  lazy val fastNer = trainFastNer()


  def getDocumentAssembler(): DocumentAssembler = {
    new DocumentAssembler()
      .setInputCol("text")
      .setOutputCol("document")
  }

  def trainStdSentenceDetector(): SentenceDetector = {
    new SentenceDetector()
      .setCustomBoundChars(Array("\n\n"))
      .setInputCols(Array("document"))
      .setOutputCol("sentence")
  }

  def trainStdTokenizer(): Tokenizer = {
    new Tokenizer()
      .setInputCols(Array("document"))
      .setOutputCol("token")
  }

  def trainFastPos(): PerceptronModel = {
    val posTagger = new PerceptronApproach()
      .setCorpus("/anc-pos-corpus/", "|")
      .setNIterations(10)
      .setInputCols("token", "document")
      .setOutputCol("pos")

    val pipeline = new Pipeline().setStages(
      Array(documentAssembler,
        stdSentenceDetector,
        stdTokenizer,
        posTagger)).fit(emptyDataset)

    pipeline.stages.last.asInstanceOf[PerceptronModel]
  }

  def trainFastNer(): NerCrfModel = {
    val nerTagger = new NerCrfApproach()
      .setInputCols("sentence", "token", "pos")
      .setLabelColumn("label")
      .setExternalDataset("eng.train")
      .setC0(2250000)
      .setRandomSeed(100)
      .setMaxEpochs(20)
      .setMinW(0.01)
      .setOutputCol("ner")
      .setEmbeddingsSource("glove.6B.100d.txt", 100, WordEmbeddingsFormat.TEXT)

    val pipeline = new Pipeline().setStages(
      Array(documentAssembler,
        stdSentenceDetector,
        stdTokenizer,
        fastPos,
        nerTagger
      )).fit(emptyDataset)

    pipeline.stages.last.asInstanceOf[NerCrfModel]
  }
}

object TrainingHelper {
  def savePipeline(s: String, someString: Some[String], libVersion: Some[Version], sparkVersion: Some[Version], pipeline: Pipeline, folder: String) = ???


  def saveModel(name: String,
                language: Option[String],
                libVersion: Option[Version],
                sparkVersion: Option[Version],
                modelWriter: MLWriter,
                folder: String
               ) = {

    // 1. Get current timestamp
    val timestamp = new Timestamp(new Date().getTime)

    // 2. Create resource metadata
    val meta = new ResourceMetadata(name, language, libVersion, sparkVersion, true, timestamp, true)

    // 3. Save model to file
    val file = Paths.get(folder, meta.key).toString
    modelWriter.save(file)

    // 4. Zip file
    val zipFile = Paths.get(folder, meta.fileName).toString
    ZipArchiveUtil.zip(file, zipFile)

    // 5. Remove original file
    FileUtils.deleteDirectory(new File(file))

    // 6. Add to metadata.json info about resource
    val metadataFile = Paths.get(folder, "metadata.json").toString
    ResourceMetadata.addMetadataToFile(metadataFile, meta)
  }
}
