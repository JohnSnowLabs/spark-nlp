package com.johnsnowlabs.benchmarks.spark

import java.io.{BufferedWriter, File, FileWriter}

import com.johnsnowlabs.nlp.annotators.common.NerTagged
import com.johnsnowlabs.nlp.datasets.CoNLL
import com.johnsnowlabs.nlp.{Annotation, SparkAccessor}
import com.johnsnowlabs.nlp.util.io.ExternalResource
import org.apache.spark.ml.PipelineModel

import scala.collection.mutable


object NerHelper {

  /**
    * Print top n Named Entity annotations
    */
  def print(annotations: Seq[Annotation], n: Int): Unit = {
    for (a <- annotations.take(n)) {
      System.out.println(s"${a.begin}, ${a.end}, ${a.result}, ${a.metadata("text")}")
    }
  }

  /**
    * Saves ner results to csv file
    * @param annotations
    * @param file
    */
  def saveNerSpanTags(annotations: Array[Array[Annotation]], file: String): Unit = {
    val bw = new BufferedWriter(new FileWriter(new File(file)))

    bw.write(s"start\tend\ttag\ttext\n")
    for (i <- 0 until annotations.length) {
      for (a <- annotations(i))
        bw.write(s"${a.begin}\t${a.end}\t${a.result}\t${a.metadata("text").replace("\n", " ")}\n")
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

  def measureExact(nerReader: CoNLL, model: PipelineModel, file: ExternalResource, printErrors: Int = 0): Unit = {
    val df = nerReader.readDataset(file, SparkAccessor.benchmarkSpark).toDF()
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
        predicted(p.result) = predicted.getOrElse(p.result, 0) + 1
      }

      for (l <- labels) {
        correct(l.result) = correct.getOrElse(l.result, 0) + 1
      }

      val correctPredictions = labels.toSet.intersect(predictions.toSet)

      for (a <- correctPredictions) {
        correctPredicted(a.result) = correctPredicted.getOrElse(a.result, 0) + 1
      }

      if (toPrintErrors > 0) {
        for (p <- predictions) {
          if (toPrintErrors > 0 && !correctPredictions.contains(p)) {
            System.out.println(s"Predicted\t${p.result}\t${p.begin}\t${p.end}\t${p.metadata("text")}")
            toPrintErrors -= 1
          }
        }

        for (p <- labels) {
          if (toPrintErrors > 0 && !correctPredictions.contains(p)) {
            System.out.println(s"Correct\t${p.result}\t${p.begin}\t${p.end}\t${p.metadata("text")}")
            toPrintErrors -= 1
          }
        }
      }
    }

    val (prec, rec, f1) = calcStat(correct.values.sum, predicted.values.sum, correctPredicted.values.sum)
    System.out.println(s"$prec\t$rec\t$f1")

    val tags = (correct.keys ++ predicted.keys ++ correctPredicted.keys).toList.distinct

    for (tag <- tags) {
      val (prec, rec, f1) = calcStat(correct.getOrElse(tag, 0), predicted.getOrElse(tag, 0), correctPredicted.getOrElse(tag, 0))
      System.out.println(s"$tag\t$prec\t$rec\t$f1")
    }
  }
}
