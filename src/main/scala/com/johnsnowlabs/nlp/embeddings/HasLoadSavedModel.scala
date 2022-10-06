package com.johnsnowlabs.nlp.embeddings
import com.johnsnowlabs.ml.tensorflow.TensorflowWrapper
import com.johnsnowlabs.nlp.util.io.{ExternalResource, ReadAs, ResourceHelper}
import org.apache.spark.sql.SparkSession

import java.io.File
import scala.reflect._

trait HasLoadSavedModel[AnnoClass <: HasDlModel[AnnoClass, FrameworkModel], FrameworkModel] {

  /** Loads a saved model from either a local or remote directory.
    *
    * TODO: Remote, potential TF/PT flexibility
    * @param folder
    *   Path to folder of the model
    * @param spark
    *   Current Spark session
    * @return
    */
  def loadSavedModel(folder: String, spark: SparkSession)(implicit
      tag: ClassTag[AnnoClass]): AnnoClass = {
    val f = new File(folder)
    val savedModel = new File(folder, "saved_model.pb")

    require(f.exists, s"Folder $folder not found")
    require(f.isDirectory, s"File $folder is not folder")
    require(savedModel.exists(), s"savedModel file saved_model.pb not found in folder $folder")

    val (wrapper, signatures) =
      TensorflowWrapper.read(folder, zipped = false, useBundle = true)

    // Create new instance using the implicitly declared ClassTag to avoid type erasure
    val annoRuntimeClass: Class[_] = tag.runtimeClass
    val annoInstance: AnnoClass = annoRuntimeClass
      .newInstance() // TODO: Some annos might have parameters?
      .asInstanceOf[AnnoClass]

    annoInstance match {
      case a: AnnoClass with HasSignature with HasVocabulary =>
        // Process Vocab
        val vocab = new File(folder + "/assets", "vocab.txt")
        require(vocab.exists(), s"Vocabulary file vocab.txt not found in folder $folder")

        val vocabResource =
          new ExternalResource(vocab.getAbsolutePath, ReadAs.TEXT, Map("format" -> "text"))
        val words = ResourceHelper.parseLines(vocabResource).zipWithIndex.toMap

        // Process Signature
        val _signatures = signatures match {
          case Some(s) => s
          case None => throw new Exception("Cannot load signature definitions from model!")
        }

        a.setVocabulary(words).setSignatures(_signatures)
      case _ => ??? // TODO: Other cases for other Annotator specific resources
    }

    annoInstance.setModelIfNotSet(spark, wrapper)
  }
}
