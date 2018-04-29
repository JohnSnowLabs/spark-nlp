
package com.johnsnowlabs.nlp.annotators.assertion.logreg

import com.johnsnowlabs.nlp.Annotation
import com.johnsnowlabs.nlp.embeddings.WordEmbeddings
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.sql.Row
import org.apache.spark.sql.functions._

import scala.collection.mutable.ArrayBuffer


/**
  * Created by jose on 24/11/17.
  */
trait Windowing extends Serializable {

  /* */
  val before : Int
  val after : Int

  val tokenizer : Tokenizer

  def wordVectors(): Option[WordEmbeddings] = None

  def tokenIndexToSubstringIndex(doc: String, s: Int, e: Int): (Int, Int) = {
    val tokens = doc.split(" ").filter(_!="")

    /* now start and end are indexes in the doc string */
    val start = tokens.slice(0, s).map(_.length).sum +
      tokens.slice(0, s).length // account for spaces
    val end = start + tokens.slice(s, e + 1).map(_.length).sum +
      tokens.slice(s, e + 1).length - 1 // account for spaces

    (start, end)
  }

  /* apply window, pad/truncate sentence according to window */
  def applyWindow(doc: String, s: Int, e: Int): (Array[String], Array[String], Array[String])  = {

    val target = doc.slice(s, e)
    val targetPart = tokenizer.tokenize(target.trim)
    val leftPart = if (s == 0) Array[String]()
    else tokenizer.tokenize(doc.slice(0, s).trim)

    val rightPart = if (e == doc.length) Array[String]()
    else tokenizer.tokenize(doc.slice(e, doc.length).trim)

    val (start, leftPadding) =
      if(leftPart.length >= before)
        (leftPart.length - before, Array[String]())
      else
        (0, Array.fill(before - leftPart.length)("empty_marker"))

    val (end, rightPadding) =
      if(targetPart.length - 1 + rightPart.length <= after)
        (rightPart.length, Array.fill(after - (targetPart.length - 1 + rightPart.length))("empty_marker"))
      else
        (after - targetPart.length, Array[String]())

    val leftContext = leftPart.slice(start, leftPart.length)
    val rightContext = rightPart.slice(0, end + 1)

    (leftPadding ++ leftContext, targetPart, rightContext ++ rightPadding)
  }

  /* apply window, pad/truncate sentence according to window */
  def applyWindow(doc: String, target: String) : (Array[String], Array[String], Array[String])= {
    val start = doc.indexOf(target)
    val end = start + target.length
    applyWindow(doc, start, end)
  }

  def applyWindow(wvectors: WordEmbeddings) (doc:String, s:Int, e:Int) : Array[Double]  = {
    val (l, t, r) = applyWindow(doc.toLowerCase, s, e)

    l.flatMap(w => normalize(wvectors.getEmbeddings(w).map(_.toDouble))) ++
      t.flatMap(w =>  normalize(wvectors.getEmbeddings(w).map(_.toDouble))) ++
      r.flatMap(w =>  normalize(wvectors.getEmbeddings(w).map(_.toDouble)))
  }

  def applyWindowUdf =
  //here 's' and 'e' are token numbers for start and end of target when split on " ". Convert to substring index first.
    udf { (doc:String, s:Int, e:Int) => {
      val (start, end) = tokenIndexToSubstringIndex(doc, s, e)
      Vectors.dense(applyWindow(wordVectors.get)(doc, start, end))
    }}

  def applyWindowUdfNerFirst(targetLabels: Array[String]) =
  // here 's' and 'e' are already substring indexes from ner annotations
    udf { (doc: String, row: Seq[Row]) =>
      var i: Option[Int] = None
      var range: Option[(Int, Int)] = None
      val annotations = row.map { r => Annotation(r) }
      annotations.zipWithIndex.filter(a => targetLabels.contains(a._1.result)).takeWhile(a => {
        if (i.isDefined) {
          if (a._2 == i.get + 1) {
            i = Some(a._2)
            range = Some((range.get._1, a._1.end))
            true
          } else {
            false
          }
        } else {
          range = Some(a._1.begin, a._1.end)
          true
        }
      })
      if (range.isDefined) {
        require(doc.slice(range.get._1, range.get._2).split(" ").length <= after,
          "NER Based assertion status failed due to targets longer than afterParam")
        Vectors.dense(applyWindow(wordVectors.get)(doc, range.get._1, range.get._2))
      }
      else
        throw new IllegalArgumentException("NER Based assertion status failed due to missing entities in nerCol")
    }

  def applyWindowUdfNerExhaustive(targetLabels: Array[String]) =
  // Reading NER annotations and calculating start-end boundaries for each contiguous entity token
    udf { (doc: String, row: Seq[Row]) => {
      val annotations = row.map { r => Annotation(r) }
      val targets = annotations.zipWithIndex.filter(a => targetLabels.contains(a._1.result)).toIterator
      val ranges = ArrayBuffer.empty[(Int, Int)]
      while (targets.hasNext) {
        val annotation = targets.next
        var range = (annotation._1.begin, annotation._1.end)
        var look = true
        while(look && targets.hasNext) {
          val nextAnnotation = targets.next
          if (nextAnnotation._2 == annotation._2 + 1)
            range = (range._1, nextAnnotation._1.end)
          else
            look = false
        }
        ranges.append(range)
      }
      if (ranges.nonEmpty) {
        require(ranges.forall(p => doc.slice(p._1, p._2).split(" ").length <= after),
          "NER Based assertion status failed due to targets longer than afterParam")
        ranges.map { r => Vectors.dense(applyWindow(wordVectors.get)(doc, r._1, r._2)) }
      }
      else
        throw new IllegalArgumentException("NER Based assertion status failed due to missing entities in nerCol")
    }}

  def l2norm(xs: Array[Double]):Double = {
    import math._
    sqrt(xs.map{ x => pow(x, 2)}.sum)
  }

  def normalize(vec: Array[Double]) : Array[Double] = {
    val norm = l2norm(vec) + 1.0
    vec.map(element => element / norm)
  }


}