/*
 * Copyright 2017-2026 John Snow Labs
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
package com.johnsnowlabs.nlp.util

import com.johnsnowlabs.nlp.{Annotation, AnnotationImage}
import org.apache.spark.sql.Row

object AnnotationRowUtils {

  def extractAnnotationRows(row: Row, columnIndex: Int): scala.collection.Seq[Row] = {
    if (row.isNullAt(columnIndex)) scala.collection.Seq.empty
    else {
      row.get(columnIndex) match {
        case rows: scala.collection.Seq[_] =>
          rows.iterator.collect { case annotationRow: Row => annotationRow }.toVector
        case rows: Array[_] =>
          rows.iterator.collect { case annotationRow: Row => annotationRow }.toVector
        case value =>
          throw new IllegalArgumentException(
            s"Expected annotation array at column $columnIndex but found ${value.getClass.getName}")
      }
    }
  }

  def annotationToRow(annotation: Annotation): Row =
    Row(
      annotation.annotatorType,
      annotation.begin,
      annotation.end,
      annotation.result,
      annotation.metadata,
      annotation.embeddings)

  def annotationsToRows(
      annotations: scala.collection.Seq[Annotation]): scala.collection.Seq[Row] =
    Option(annotations).getOrElse(scala.collection.Seq.empty).map(annotationToRow)

  def annotationImageToRow(annotationImage: AnnotationImage): Row =
    Row(
      annotationImage.annotatorType,
      annotationImage.origin,
      annotationImage.height,
      annotationImage.width,
      annotationImage.nChannels,
      annotationImage.mode,
      annotationImage.result,
      annotationImage.metadata,
      annotationImage.text)

  def annotationImagesToRows(
      annotationImages: scala.collection.Seq[AnnotationImage]): scala.collection.Seq[Row] =
    Option(annotationImages).getOrElse(scala.collection.Seq.empty).map(annotationImageToRow)
}
