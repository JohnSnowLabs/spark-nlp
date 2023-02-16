package com.johnsnowlabs.nlp.annotators.similarity

import com.johnsnowlabs.nlp.{AnnotatorApproach, ParamsAndFeaturesWritable}
import org.apache.spark.ml.PipelineModel
import org.apache.spark.sql.Dataset


class DocumentSimilarityRankerApproach(override val uid: String)
extends AnnotatorApproach[DocumentSimilarityRankerModel]
with ParamsAndFeaturesWritable
{
  override val description: AnnotatorType = ???

  override def train(dataset: Dataset[_], recursivePipeline: Option[PipelineModel]): DocumentSimilarityRankerModel = ???

  override val outputAnnotatorType: AnnotatorType = ???
  /** Annotator reference id. Used to identify elements in metadata or to refer to this annotator
   * type
   */
  override val inputAnnotatorTypes: Array[AnnotatorType] = ???
}
