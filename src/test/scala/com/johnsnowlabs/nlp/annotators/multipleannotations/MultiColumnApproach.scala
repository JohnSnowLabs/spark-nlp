package com.johnsnowlabs.nlp.annotators.multipleannotations

import com.johnsnowlabs.nlp.{AnnotatorApproach, HasMultipleInputAnnotationCols}
import com.johnsnowlabs.nlp.AnnotatorType.{CHUNK, DOCUMENT}
import org.apache.spark.ml.PipelineModel
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.sql.Dataset


class MultiColumnApproach(override val uid: String) extends AnnotatorApproach[MultiColumnsModel] with HasMultipleInputAnnotationCols{

  def this() = this(Identifiable.randomUID("multiplecolums"))
  override val description: String = "Example multiple columns"

  /**
   * Input annotator types: CHUNK
   *
   * @group anno
   */
  override val outputAnnotatorType: AnnotatorType = DOCUMENT
  /**
   * Output annotator types: CHUNK, CHUNK
   *
   * @group anno
   */
  override val inputAnnotatorType: AnnotatorType = DOCUMENT


  /** whether to merge overlapping matched chunks. Defaults to true
   *
   * @group param
   * */

  override def train(dataset: Dataset[_], recursivePipeline: Option[PipelineModel]): MultiColumnsModel = {

    new MultiColumnsModel().setInputCols($(inputCols)).setOutputCol($(outputCol))
  }


}
