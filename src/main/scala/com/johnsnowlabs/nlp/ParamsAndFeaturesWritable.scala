package com.johnsnowlabs.nlp

import org.apache.spark.ml.param.Params
import org.apache.spark.ml.util.{DefaultParamsWritable, MLWriter}
import org.apache.spark.sql.SparkSession

class FeaturesWriter[T](annotatorWithFeatures: HasFeatures, baseWriter: MLWriter, onWritten: (String, SparkSession) => Unit)
  extends MLWriter with HasFeatures {

  override protected def saveImpl(path: String): Unit = {
    baseWriter.save(path)

    for (feature <- annotatorWithFeatures.features) {
      feature.serializeInfer(sparkSession, path, feature.name, feature.getOrDefault)
    }

    onWritten(path, sparkSession)

  }
}

trait ParamsAndFeaturesWritable extends DefaultParamsWritable with Params with HasFeatures {

  def onWrite(path: String, spark: SparkSession): Unit = {}

  override def write: MLWriter = new FeaturesWriter(
    this,
    super.write,
    (path: String, spark: SparkSession) => onWrite(path, spark)
  )

}
