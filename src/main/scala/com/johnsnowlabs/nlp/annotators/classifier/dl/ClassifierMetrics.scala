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

package com.johnsnowlabs.nlp.annotators.classifier.dl

import com.johnsnowlabs.ml.tensorflow.Logging
import org.apache.spark.ml.util.Identifiable

trait ClassifierMetrics extends Logging {

  def calcStat(tp: Int, fp: Int, fn: Int): (Float, Float, Float) = {
    val precision = tp.toFloat / (tp.toFloat + fp.toFloat)
    val recall = tp.toFloat / (tp.toFloat + fn.toFloat)
    val f1 = 2 * ((precision * recall) / (precision + recall))

    (
      if (precision.isNaN) 0f else precision,
      if (recall.isNaN) 0f else recall,
      if (f1.isNaN) 0 else f1)
  }

  def aggregatedMetrics(
      labels: Seq[String],
      truePositives: Map[String, Int],
      falsePositives: Map[String, Int],
      falseNegatives: Map[String, Int],
      extended: Boolean,
      enableOutputLogs: Boolean,
      outputLogsPath: String,
      uuid: String = Identifiable.randomUID("ClassifierMetrics")): (Float, Float) = {

    val totalTruePositives = truePositives.values.sum
    val totalFalsePositives = falsePositives.values.sum
    val totalFalseNegatives = falseNegatives.values.sum
    val (precision, recall, f1) =
      calcStat(totalTruePositives, totalFalsePositives, totalFalseNegatives)

    val maxLabelLength = labels.maxBy(_.length).length
    val headLabelSpace = computeLabelSpace(maxLabelLength)

    if (extended) {
      println(s"label$headLabelSpace tp\t fp\t fn\t prec\t rec\t f1")
      outputLog(
        s"label$headLabelSpace tp\t fp\t fn\t prec\t rec\t f1",
        uuid,
        enableOutputLogs,
        outputLogsPath)
    }

    var totalPercByClass, totalRecByClass = 0f
    for (label <- labels) {
      val tp = truePositives.getOrElse(label, 0)
      val fp = falsePositives.getOrElse(label, 0)
      val fn = falseNegatives.getOrElse(label, 0)
      val (precision, recall, f1) = calcStat(tp, fp, fn)

      if (extended) {

        val labelOutput = computeLabelOutput(label, maxLabelLength, headLabelSpace.length)

        println(s"$labelOutput $tp\t $fp\t $fn\t $precision\t $recall\t $f1")
        outputLog(
          s"$labelOutput $tp\t $fp\t $fn\t $precision\t $recall\t $f1",
          uuid,
          enableOutputLogs,
          outputLogsPath)
      }

      totalPercByClass = totalPercByClass + precision
      totalRecByClass = totalRecByClass + recall
    }
    val macroPrecision = totalPercByClass / labels.length
    val macroRecall = totalRecByClass / labels.length
    val macroF1 = 2 * ((macroPrecision * macroRecall) / (macroPrecision + macroRecall))

    if (extended) {
      println(
        s"tp: $totalTruePositives fp: $totalFalsePositives fn: $totalFalseNegatives labels: ${labels.length}")
      outputLog(
        s"tp: $totalTruePositives fp: $totalFalsePositives fn: $totalFalseNegatives labels: ${labels.length}",
        uuid,
        enableOutputLogs,
        outputLogsPath)
    }
    // ex: Precision = P1+P2/2
    println(s"Macro-average\t prec: $macroPrecision, rec: $macroRecall, f1: $macroF1")
    outputLog(
      s"Macro-average\t prec: $macroPrecision, rec: $macroRecall, f1: $macroF1",
      uuid,
      enableOutputLogs,
      outputLogsPath)
    // ex: Precision =  TP1+TP2/TP1+TP2+FP1+FP2
    println(s"Micro-average\t prec: $precision, recall: $recall, f1: $f1")
    outputLog(
      s"Micro-average\t prec: $precision, recall: $recall, f1: $f1",
      uuid,
      enableOutputLogs,
      outputLogsPath)

    (f1, macroF1)
  }

  private def computeLabelSpace(maxLabelLength: Int): String = {
    val spaceToFill = maxLabelLength - "label".length
    val labelSpace = if (spaceToFill <= 0) {
      Array.fill("label".length + 2)(" ").mkString("")
    } else Array.fill(spaceToFill + 2)(" ").mkString("")

    labelSpace
  }

  private def computeLabelOutput(
      label: String,
      maxLabelLength: Int,
      headLabelLength: Int): String = {

    if (label.length > "label".length) {
      val spaceToFill = maxLabelLength - label.length + 2
      val currentLabelSpace = Array.fill(spaceToFill)(" ").mkString("")
      s"$label$currentLabelSpace"
    } else {
      val spaceToFill = "label".length - label.length
      val currentLabelSpace = Array.fill(spaceToFill + headLabelLength)(" ").mkString("")
      s"$label$currentLabelSpace"
    }

  }

}
