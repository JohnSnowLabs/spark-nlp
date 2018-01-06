package com.johnsnowlabs.nlp

import org.apache.spark.ml.param.Params
import org.apache.spark.ml.util.{DefaultParamsWritable, MLWriter}

class FeaturesWriter[T](annotatorWithFeatures: HasFeatures, baseWriter: MLWriter) extends MLWriter with HasFeatures {

  override protected def saveImpl(path: String): Unit = {
    baseWriter.save(path)

    for (feature <- annotatorWithFeatures.features) {
      feature.serializeInfer(sparkSession, path, feature.name, feature.getValue)
    }
  }
}

trait ParamsAndFeaturesWritable extends DefaultParamsWritable with Params with HasFeatures {

  override def write: MLWriter = new FeaturesWriter(this, super.write)

}
