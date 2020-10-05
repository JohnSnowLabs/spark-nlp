package com.johnsnowlabs.nlp

import org.apache.spark.ml.Model
import org.apache.spark.ml.param.IntParam
import org.apache.spark.sql.Row
import org.apache.spark.sql.catalyst.expressions.GenericRowWithSchema

trait HasBatchedAnnotate[M <: Model[M]] {

  this: RawAnnotator[M] =>

  /** Size of every batch.
    *
    * @group param
    * */
  val batchSize = new IntParam(this, "batchSize", "Size of every batch.")

  /** Size of every batch.
    *
    * @group setParam
    * */
  def setBatchSize(size: Int): this.type = set(this.batchSize, size)

  /** Size of every batch.
    *
    * @group getParam
    * */
  def getBatchSize: Int = $(batchSize)

  def batchProcess(rows: Iterator[_]): Iterator[Row] = {
    rows.grouped(getBatchSize).flatMap { case batchedRows: Seq[Row] =>
      val inputAnnotations = getInputCols.flatMap(inputCol => {
        batchedRows.map(_.getAs[Seq[Row]](inputCol).map(Annotation(_)))
      })
      val outputAnnotations = batchAnnotate(inputAnnotations)
      batchedRows.zip(outputAnnotations).map { case (row, annotations) =>
        row.toSeq ++ Array(annotations.map(a => Row(a.productIterator.toSeq:_*)))
      }
    }.map(Row.fromSeq)
  }

  def batchAnnotate(batchedAnnotations: Array[Seq[Annotation]]): Seq[Seq[Annotation]]

}
