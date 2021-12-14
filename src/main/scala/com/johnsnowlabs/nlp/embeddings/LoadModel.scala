package com.johnsnowlabs.nlp.embeddings

import com.johnsnowlabs.ml.pytorch.PytorchWrapper
import com.johnsnowlabs.ml.tensorflow.TensorflowWrapper
import com.johnsnowlabs.nlp.util.io.{ExternalResource, ReadAs, ResourceHelper}
import org.apache.spark.sql.SparkSession

import java.io.File

trait LoadModel[T] {

  def loadSavedModel(tfModelPath: String, spark: SparkSession): T = {
    val (tfWrapper, signatures) = loadSavedTensorflowModel(tfModelPath)
    val vocabulary = loadVocabulary(tfModelPath, "tensorflow")

    createEmbeddingsFromTensorflow(tfWrapper, signatures, vocabulary, spark)
  }

  def createEmbeddingsFromTensorflow(tfWrapper: TensorflowWrapper, signatures: Map[String, String],
                                     vocabulary: Map[String, Int], spark: SparkSession): T

  def loadSavedTensorflowModel(tfModelPath: String): (TensorflowWrapper, Map[String, String]) = {

    validateModelPath(tfModelPath)
    val savedModel = new File(tfModelPath, "saved_model.pb")
    require(savedModel.exists(), s"savedModel file saved_model.pb not found in folder $tfModelPath")

    val (tfWrapper, signatures) = TensorflowWrapper.read(tfModelPath, zipped = false, useBundle = true)

    val _signatures = signatures match {
      case Some(s) => s
      case None => throw new Exception("Cannot load signature definitions from model!")
    }

    (tfWrapper, _signatures)
  }

  def loadTorchScriptModel(torchModelPath: String, spark: SparkSession): T = {
    validateModelPath(torchModelPath)
    val pytorchWrapper = PytorchWrapper(torchModelPath)
    val vocabulary = loadVocabulary(torchModelPath, "pytorch")

    createEmbeddingsFromPytorch(pytorchWrapper, vocabulary, spark)
  }

  def createEmbeddingsFromPytorch(pytorchWrapper: PytorchWrapper, vocabulary: Map[String, Int], spark: SparkSession): T

  private def validateModelPath(modelPath: String): Unit = {
    val modelFile = new File(modelPath)
    require(modelFile.exists, s"Folder $modelPath not found")
    require(modelFile.isDirectory, s"File $modelPath is not folder")
  }

  def loadVocabulary(modelPath: String, engine: String): Map[String, Int] = {

    val vocab = if(engine == "pytorch") new File(modelPath, "vocab.txt") else new File(modelPath + "/assets", "vocab.txt")
    require(vocab.exists(), s"Vocabulary file vocab.txt not found in folder $modelPath")

    val vocabResource = new ExternalResource(vocab.getAbsolutePath, ReadAs.TEXT, Map("format" -> "text"))
    val words = ResourceHelper.parseLines(vocabResource).zipWithIndex.toMap

    words
  }

}
