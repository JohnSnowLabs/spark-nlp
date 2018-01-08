package com.johnsnowlabs.nlp

import com.johnsnowlabs.nlp.embeddings.ModelWithWordEmbeddings
import org.apache.spark.ml.util.{DefaultParamsReadable, MLReader}

class FeaturesReader[T <: HasFeatures](baseReader: MLReader[T]) extends MLReader[T] {

  override def load(path: String): T = {

    val instance = baseReader.load(path)

    for (feature <- instance.features) {
      val value = feature.deserialize(sparkSession, path, feature.name)
      feature.setValue(value)
    }

    instance match {
      case m: ModelWithWordEmbeddings[_] => m.deserializeEmbeddings(path, sparkSession.sparkContext)
      case _ =>
    }

    instance
  }
}

trait ParamsAndFeaturesReadable[T <: HasFeatures] extends DefaultParamsReadable[T] {
  override def read: MLReader[T] = new FeaturesReader(super.read)
}
