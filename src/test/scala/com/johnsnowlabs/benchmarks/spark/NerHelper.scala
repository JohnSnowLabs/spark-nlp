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

package com.johnsnowlabs.benchmarks.spark

import java.io.{BufferedWriter, File, FileWriter}

import com.johnsnowlabs.nlp.annotators.common.NerTagged
import com.johnsnowlabs.nlp.training.CoNLL
import com.johnsnowlabs.nlp.{Annotation, SparkAccessor}
import com.johnsnowlabs.nlp.util.io.ExternalResource
import org.apache.spark.ml.PipelineModel

import scala.collection.mutable

object NerHelper {

  /** Print top n Named Entity annotations */
  def print(annotations: Seq[Annotation], n: Int): Unit = {
    for (a <- annotations.take(n)) {
      System.out.println(s"${a.begin}, ${a.end}, ${a.result}, ${a.metadata("text")}")
    }
  }

  /** Saves ner results to csv file
    * @param annotations
    * @param file
    */
  def saveNerSpanTags(annotations: Array[Array[Annotation]], file: String): Unit = {
    val bw = new BufferedWriter(new FileWriter(new File(file)))

    bw.write(s"start\tend\ttag\ttext\n")
    for (i <- 0 until annotations.length) {
      for (a <- annotations(i))
        bw.write(
          s"${a.begin}\t${a.end}\t${a.result}\t${a.metadata("entity").replace("\n", " ")}\n")
    }
    bw.close()
  }

  def calcStat(correct: Int, predicted: Int, predictedCorrect: Int): (Float, Float, Float) = {
    // prec = (predicted & correct) / predicted
    // rec = (predicted & correct) / correct
    val prec = predictedCorrect.toFloat / predicted
    val rec = predictedCorrect.toFloat / correct
    val f1 = 2 * prec * rec / (prec + rec)

    (prec, rec, f1)
  }

  def measureExact(
      nerReader: CoNLL,
      model: PipelineModel,
      file: ExternalResource,
      printErrors: Int = 0): Unit = {
    val df = nerReader.readDataset(SparkAccessor.benchmarkSpark, file.path).toDF()
    val transformed = model.transform(df)
    val rows = transformed.select("ner_span", "label_span").collect()

    val correctPredicted = mutable.Map[String, Int]()
    val predicted = mutable.Map[String, Int]()
    val correct = mutable.Map[String, Int]()
    var toPrintErrors = printErrors

    for (row <- rows) {

      val predictions = NerTagged.getAnnotations(row, 0).filter(a => a.result != "O")
      val labels = NerTagged.getAnnotations(row, 1).filter(a => a.result != "O")

      for (p <- predictions) {
        val tag = p.metadata("entity")
        predicted(tag) = predicted.getOrElse(tag, 0) + 1
      }

      for (l <- labels) {
        val tag = l.metadata("entity")
        correct(tag) = correct.getOrElse(tag, 0) + 1
      }

      val correctPredictions = labels.toSet.intersect(predictions.toSet)

      for (a <- correctPredictions) {
        val tag = a.metadata("entity")
        correctPredicted(tag) = correctPredicted.getOrElse(tag, 0) + 1
      }

      if (toPrintErrors > 0) {
        for (p <- predictions) {
          if (toPrintErrors > 0 && !correctPredictions.contains(p)) {
            System.out.println(
              s"Predicted\t${p.result}\t${p.begin}\t${p.end}\t${p.metadata("text")}")
            toPrintErrors -= 1
          }
        }

        for (p <- labels) {
          if (toPrintErrors > 0 && !correctPredictions.contains(p)) {
            System.out.println(
              s"Correct\t${p.result}\t${p.begin}\t${p.end}\t${p.metadata("text")}")
            toPrintErrors -= 1
          }
        }
      }
    }

    val (prec, rec, f1) =
      calcStat(correct.values.sum, predicted.values.sum, correctPredicted.values.sum)
    System.out.println(s"$prec\t$rec\t$f1")

    val tags = (correct.keys ++ predicted.keys ++ correctPredicted.keys).toList.distinct

    for (tag <- tags) {
      val (prec, rec, f1) = calcStat(
        correct.getOrElse(tag, 0),
        predicted.getOrElse(tag, 0),
        correctPredicted.getOrElse(tag, 0))
      System.out.println(s"$tag\t$prec\t$rec\t$f1")
    }
  }
}
