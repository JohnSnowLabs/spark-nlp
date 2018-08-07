
package com.johnsnowlabs.nlp.annotators.assertion.logreg

import com.johnsnowlabs.nlp.Annotation
import com.johnsnowlabs.nlp.embeddings.WordEmbeddings
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.sql.Row
import org.apache.spark.sql.functions._

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
    udf { (documents:Seq[Row], s:Int, e:Int) => {
      /** NOTE: Yes, this only works with one sentence per row, start end applies only to first */
      val doc = Annotation(documents.head).result
      val (start, end) = tokenIndexToSubstringIndex(doc, s, e)
      Vectors.dense(applyWindow(wordVectors().get)(doc, start, end))
    }}

  private case class IndexedChunk(sentence: String, chunkBegin: Int, chunkEnd: Int)

  def applyWindowUdfNerExhaustive =
  // Reading NER annotations and calculating start-end boundaries for each contiguous entity token
    udf { (documents: Seq[Row], chunks: Seq[Row]) => {
      //println(s"all documents: ${documents.map(Annotation(_).result).mkString(", ")}")
      //println(s"all chunks: ${chunks.map(Annotation(_).result).mkString(", ")}")

      var lastIC: Option[IndexedChunk] = None

      val indexed = documents.map(Annotation(_).result).zipAll(chunks.map(Annotation(_).result), "", "")
        .map { case (doc, chunk) =>
            if (chunk.isEmpty) {
              IndexedChunk("", 0, 0)
            } else if (doc.isEmpty) {
              /** More than one chunk per document*/
              lastIC.get
            } else {
              require(doc.contains(chunk), s"Chunk: $chunk is not a substring of document: $doc")
              val index = doc.indexOf(chunk)
              var tokenIndexBegin = 0
              for (i <- 0 until index) {
                if (doc(i) == ' ')
                  tokenIndexBegin += 1
              }
              val tokenIndexEnd = tokenIndexBegin + chunk.split(" ").length - 1
              val ic = IndexedChunk(doc, tokenIndexBegin, tokenIndexEnd)
              lastIC = Some(ic)
              ic
          }
        }

      indexed.map ( r => Vectors.dense(applyWindow(wordVectors().get)(r.sentence, r.chunkBegin, r.chunkEnd)) )

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