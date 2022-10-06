package com.johnsnowlabs.nlp.embeddings
import com.johnsnowlabs.ml.tensorflow.TensorflowWrapper
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.sql.SparkSession

trait HasDlModel[AnnoClass, FrameworkModel] {
  protected var _model: Option[Broadcast[FrameworkModel]] = None

  def getModelIfNotSet: FrameworkModel = _model.get.value

  // TODO: Might be able to put implementation here with pattern matching traits
  def setModelIfNotSet(spark: SparkSession, tensorflow: TensorflowWrapper): AnnoClass

}
