package com.johnsnowlabs.nlp

import org.apache.spark.ml.util.Identifiable
import org.apache.spark.ml.{Estimator, Pipeline, PipelineModel, Transformer}
import org.apache.spark.sql.Dataset

import scala.collection.mutable.ListBuffer

class RecursivePipeline(override val uid: String) extends Pipeline {

  def this() = this(Identifiable.randomUID("RECURSIVE_PIPELINE"))

  /**Workaround to PipelineModel being private in Spark*/
  private def createPipeline(dataset: Dataset[_], uid: String, transformers: Array[Transformer]) = {
    new Pipeline().setStages(transformers).fit(dataset)
  }


  /** Has to behave as of spark 2.x.x */
  override def fit(dataset: Dataset[_]): PipelineModel = {
    transformSchema(dataset.schema, logging = true)
    val theStages = $(stages)
    var indexOfLastEstimator = -1
    theStages.view.zipWithIndex.foreach { case (stage, index) =>
      stage match {
        case _: Estimator[_] =>
          indexOfLastEstimator = index
        case _ =>
      }
    }
    var curDataset = dataset
    val transformers = ListBuffer.empty[Transformer]
    theStages.view.zipWithIndex.foreach { case (stage, index) =>
      if (index <= indexOfLastEstimator) {
        val transformer = stage match {
          case estimator: HasRecursiveFit[_] =>
            estimator.recursiveFit(curDataset, createPipeline(curDataset, uid, transformers.toArray))
          case estimator: Estimator[_] =>
            estimator.fit(curDataset)
          case t: Transformer =>
            t
          case _ =>
            throw new IllegalArgumentException(
              s"Does not support stage $stage of type ${stage.getClass}")
        }
        if (index < indexOfLastEstimator) {
          curDataset = transformer.transform(curDataset)
        }
        transformers += transformer
      } else {
        transformers += stage.asInstanceOf[Transformer]
      }
    }

    createPipeline(curDataset, uid, transformers.toArray).setParent(this)
  }

}
