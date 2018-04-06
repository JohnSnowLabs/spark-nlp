package com.johnsnowlabs.nlp

import org.apache.spark.ml.util.{DefaultParamsReadable, MLReader}
import org.apache.spark.sql.SparkSession

import scala.collection.mutable.ArrayBuffer

class FeaturesReader[T <: HasFeatures](baseReader: MLReader[T], onRead: (T, String, SparkSession) => Unit) extends MLReader[T] {

  override def load(path: String): T = {

    val instance = baseReader.load(path)

    for (feature <- instance.features) {
      val value = feature.deserialize(sparkSession, path, feature.name)
      feature.setValue(value)
    }

    onRead(instance, path, sparkSession)

    instance
  }
}

trait ParamsAndFeaturesReadable[T <: HasFeatures] extends DefaultParamsReadable[T] {

  private val readers = ArrayBuffer.empty[(T, String, SparkSession) => Unit]

  private def onRead(instance: T, path: String, session: SparkSession): Unit = {
    for (reader <- readers) {
      reader(instance, path, session)
    }
  }

  def addReader(reader: (T, String, SparkSession) => Unit): Unit = {
    readers.append(reader)
  }

  override def read: MLReader[T] = new FeaturesReader(
    super.read,
    (instance: T, path: String, spark: SparkSession) => onRead(instance, path, spark)
  )
}
