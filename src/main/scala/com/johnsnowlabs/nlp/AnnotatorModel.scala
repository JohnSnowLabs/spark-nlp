/*
 * Copyright 2017-2022 John Snow Labs
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.johnsnowlabs.nlp

import com.johnsnowlabs.nlp.util.{AnnotationRowUtils, SparkNlpConfig}
import com.johnsnowlabs.util.Version
import org.apache.spark.ml.{Model, PipelineModel}
import org.apache.spark.sql.catalyst.encoders.ExpressionEncoder
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.{StructField, StructType}
import org.apache.spark.sql.{DataFrame, Dataset, Row}

/** This trait implements logic that applies nlp using Spark ML Pipeline transformers Should
  * strongly change once UsedDefinedTypes are allowed
  * https://issues.apache.org/jira/browse/SPARK-7768
  */
abstract class AnnotatorModel[M <: Model[M]] extends RawAnnotator[M] with CanBeLazy {

  /** internal types to show Rows as a relevant StructType Should be deleted once Spark releases
    * UserDefinedTypes to @developerAPI
    */
  protected type AnnotationContent = Seq[Row]

  protected def beforeAnnotate(dataset: Dataset[_]): Dataset[_] = dataset

  protected def afterAnnotate(dataset: DataFrame): DataFrame = dataset

  private def isSpark4OrNewer(dataset: Dataset[_]): Boolean =
    Version.parse(dataset.sparkSession.version).toFloat >= 4.0f

  private def outputSchemaWithReplacement(inputSchema: StructType): StructType = {
    val outputField = StructField(getOutputCol, Annotation.arrayType, nullable = true)
    val outputIndex = inputSchema.fieldNames.indexOf(getOutputCol)

    if (outputIndex >= 0) StructType(inputSchema.fields.updated(outputIndex, outputField))
    else StructType(inputSchema.fields :+ outputField)
  }

  private def transformSimpleAnnotateWithRows(
      inputDataset: Dataset[_],
      outputSchema: StructType,
      withAnnotate: HasSimpleAnnotate[M],
      recursivePipeline: Option[PipelineModel]): DataFrame = {
    val inputDataFrame = inputDataset.toDF()
    val inputIndexes = getInputCols.map(inputDataFrame.schema.fieldIndex)
    val outputIndex = inputDataFrame.schema.fieldNames.indexOf(getOutputCol)

    implicit val encoder: ExpressionEncoder[Row] =
      SparkNlpConfig.getEncoder(inputDataFrame, outputSchema)

    val mappedDataFrame = inputDataFrame.mapPartitions { rows =>
      rows.map { row =>
        val annotationProperties = inputIndexes.map { inputIndex =>
          AnnotationRowUtils.extractAnnotationRows(row, inputIndex).toVector
        }.toVector

        val outputAnnotations = this match {
          case recursiveAnnotator: HasRecursiveTransform[M] =>
            recursiveAnnotator.recAnnotateColumnGroups(
              annotationProperties,
              recursivePipeline.get)
          case _ =>
            withAnnotate.annotateColumnGroups(annotationProperties)
        }

        val outputRows = AnnotationRowUtils.annotationsToRows(outputAnnotations).toVector
        val outputValues =
          if (outputIndex >= 0) row.toSeq.updated(outputIndex, outputRows)
          else row.toSeq :+ outputRows

        Row.fromSeq(outputValues)
      }
    }

    val withInputMetadata = inputDataFrame.schema.fields
      .filter(field => mappedDataFrame.columns.contains(field.name))
      .foldLeft(mappedDataFrame)((dataFrame, field) => {
        dataFrame.withColumn(field.name, dataFrame.col(field.name).as(field.name, field.metadata))
      })

    withInputMetadata.withColumn(getOutputCol, wrapColumnMetadata(col(getOutputCol)))
  }

  protected def _transform(
      dataset: Dataset[_],
      recursivePipeline: Option[PipelineModel]): DataFrame = {
    require(
      validate(dataset.schema),
      s"Wrong or missing inputCols annotators in $uid.\n" +
        msgHelper(dataset.schema) +
        s"\nMake sure such annotators exist in your pipeline, " +
        s"with the right output names and that they have following annotator types: " +
        s"${inputAnnotatorTypes.mkString(", ")}")

    val inputDataset = beforeAnnotate(dataset)
    val newStructType = inputDataset.schema.add(getOutputCol, Annotation.arrayType)
    val processedDataset = {
      this match {
        case withAnnotate: HasSimpleAnnotate[M] =>
          if (isSpark4OrNewer(inputDataset)) {
            transformSimpleAnnotateWithRows(
              inputDataset,
              outputSchemaWithReplacement(inputDataset.schema),
              withAnnotate,
              recursivePipeline)
          } else {
            inputDataset.withColumn(
              getOutputCol,
              wrapColumnMetadata({
                this match {
                  case a: HasRecursiveTransform[M] =>
                    a.dfRecAnnotate(recursivePipeline.get)(
                      array(getInputCols.map(c => inputDataset.col(c)): _*))
                  case _ =>
                    withAnnotate.dfAnnotate(array(getInputCols.map(c => inputDataset.col(c)): _*))
                }
              }))
          }
        case withBatchAnnotate: HasBatchedAnnotate[M] =>
          implicit val encoder: ExpressionEncoder[Row] =
            SparkNlpConfig.getEncoder(inputDataset, newStructType)
          val processedDataFrame = inputDataset.mapPartitions(partition => {
            withBatchAnnotate.batchProcess(partition)
          })

          /** Put back column metadata from `inputDataset` after destructive mapPartitions */
          val dfWithMetadata = inputDataset.schema.fields
            .foldLeft(processedDataFrame)((dataFrame, field) => {
              dataFrame
                .withColumn(field.name, dataFrame.col(field.name).as(field.name, field.metadata))
            })
            .withColumn(getOutputCol, wrapColumnMetadata(col(getOutputCol)))
          dfWithMetadata

        case withBatchAnnotateImage: HasBatchedAnnotateImage[M] =>
          implicit val encoder: ExpressionEncoder[Row] =
            SparkNlpConfig.getEncoder(inputDataset, newStructType)
          val processedDataFrame = inputDataset.mapPartitions(partition => {
            withBatchAnnotateImage.batchProcess(partition)
          })

          /** Put back column metadata from `inputDataset` after destructive mapPartitions */
          val dfWithMetadata = inputDataset.schema.fields
            .foldLeft(processedDataFrame)((dataFrame, field) => {
              dataFrame
                .withColumn(field.name, dataFrame.col(field.name).as(field.name, field.metadata))
            })
            .withColumn(getOutputCol, wrapColumnMetadata(col(getOutputCol)))
          dfWithMetadata

        case withBatchAnnotateAudio: HasBatchedAnnotateAudio[M] =>
          implicit val encoder: ExpressionEncoder[Row] =
            SparkNlpConfig.getEncoder(inputDataset, newStructType)
          val processedDataFrame = inputDataset.mapPartitions(partition => {
            withBatchAnnotateAudio.batchProcess(partition)
          })

          /** Put back column metadata from `inputDataset` after destructive mapPartitions */
          val dfWithMetadata = inputDataset.schema.fields
            .foldLeft(processedDataFrame)((dataFrame, field) => {
              dataFrame
                .withColumn(field.name, dataFrame.col(field.name).as(field.name, field.metadata))
            })
            .withColumn(getOutputCol, wrapColumnMetadata(col(getOutputCol)))
          dfWithMetadata

        case withBatchAnnotateTextImage: HasBatchedAnnotateTextImage[M] =>
          implicit val encoder: ExpressionEncoder[Row] =
            SparkNlpConfig.getEncoder(inputDataset, newStructType)
          val processedDataFrame = inputDataset.mapPartitions(partition => {
            withBatchAnnotateTextImage.batchProcess(partition)
          })

          // TODO: Do we really need to repeat this in every case?
          /** Put back column metadata from `inputDataset` after destructive mapPartitions */
          val dfWithMetadata = inputDataset.schema.fields
            .foldLeft(processedDataFrame)((dataFrame, field) => {
              dataFrame
                .withColumn(field.name, dataFrame.col(field.name).as(field.name, field.metadata))
            })
            .withColumn(getOutputCol, wrapColumnMetadata(col(getOutputCol)))
          dfWithMetadata
      }
    }

    afterAnnotate(processedDataset)
  }

  /** Given requirements are met, this applies ML transformation within a Pipeline or stand-alone
    * Output annotation will be generated as a new column, previous annotations are still
    * available separately metadata is built at schema level to record annotations structural
    * information outside its content
    *
    * @param dataset
    *   [[Dataset[Row]]]
    * @return
    */
  override final def transform(dataset: Dataset[_]): DataFrame = {
    _transform(dataset, None)
  }

}
