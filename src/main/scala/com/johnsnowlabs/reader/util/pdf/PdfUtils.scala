/*
 * Copyright 2017-2025 John Snow Labs
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
package com.johnsnowlabs.reader.util.pdf

import com.johnsnowlabs.nlp.IAnnotation

trait PdfUtils {
  val MAX_CHARACTER_BEFORE_HEADER = 1000

  def checkAndFixPdf(content: Array[Byte]): Array[Byte] = {
    val pdfStartIndex = new String(
      content.slice(0, Math.min(MAX_CHARACTER_BEFORE_HEADER, content.length))).indexOf("%PDF")
    if (pdfStartIndex == -1) throw new RuntimeException("Pdf document is not valid")
    val validContent = content.slice(pdfStartIndex, content.length)
    validContent
  }

  implicit class exceptionUtils(existingException: String) {
    def concatException(newException: String): String = {
      newException match {
        case null => existingException
        case _ => {
          existingException match {
            case null => newException
            case _ => existingException + " " + newException
          }
        }
      }
    }
  }

  implicit class ChainException(lightRecord: Map[String, Seq[IAnnotation]]) {
    def chainExceptions(
        e: IAnnotation,
        exceptionKey: String = "exception"): Map[String, Seq[IAnnotation]] = {
      // chain exceptions
      val chainedExceptions = lightRecord.getOrElse("exception", Seq.empty) :+ e
      lightRecord.updated(exceptionKey, chainedExceptions)
    }
    def chainExceptions(e: String): Map[String, Seq[IAnnotation]] = chainExceptions(
      PdfPipelineException(e))
  }

}

case class PdfPipelineException(message: String, source: String = "pdf_pipeline_exception")
    extends IAnnotation {
  override def annotatorType: String = source
  def asString = s"$source::$message"
}
