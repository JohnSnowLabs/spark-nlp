package com.johnsnowlabs.nlp.annotators.assertion.dl

import com.johnsnowlabs.nlp.annotators.ner.dl.NerDLModel
import com.johnsnowlabs.nlp.embeddings.ApproachWithWordEmbeddings
import org.apache.spark.ml.PipelineModel
import org.apache.spark.sql.Dataset

/**
  * Created by jose on 14/03/18.
  */
class AssertionDLApproach (override val uid: String)
  extends ApproachWithWordEmbeddings[AssertionDLApproach, AssertionDLModel]{
  override val description: String = ???

  override def train(dataset: Dataset[_], recursivePipeline: Option[PipelineModel]): AssertionDLModel = ???

  override val requiredAnnotatorTypes: Array[String] = ???
  override val annotatorType: AnnotatorType = ???
}
