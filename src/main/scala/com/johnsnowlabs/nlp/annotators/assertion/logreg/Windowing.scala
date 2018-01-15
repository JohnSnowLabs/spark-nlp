package com.johnsnowlabs.nlp.annotators.assertion.logreg


import com.johnsnowlabs.nlp.annotators.datasets.I2b2AnnotationAndText
import com.johnsnowlabs.nlp.embeddings.WordEmbeddings
import org.apache.spark.sql.functions._
import org.apache.spark.sql.catalyst.expressions.GenericRowWithSchema
import org.apache.spark.sql.expressions.UserDefinedFunction

import scala.collection.mutable


/**
  * Created by jose on 24/11/17.
  */

/* Type class to handle multiple Vectors versions */
trait VectorsType[T] {
  /* create a vector from an array */
  def dense(array:Array[Double]):T
}

object mllib {
  import org.apache.spark.mllib.linalg.Vector
  import org.apache.spark.mllib.linalg.Vectors
  object vectors extends VectorsType[Vector] {
    override def dense(array: Array[Double]) = Vectors.dense(array)
  }
}

object ml {
  import org.apache.spark.ml.linalg.Vector
  import org.apache.spark.ml.linalg.Vectors
  object vectors extends VectorsType[Vector] {
    override def dense(array: Array[Double]) = Vectors.dense(array)
  }
}

trait Windowing extends Serializable {

  /* */
  val before : Int
  val after : Int

  val tokenizer : Tokenizer

  lazy val wordVectors: Option[WordEmbeddings] = None

  /* apply window, pad/truncate sentence according to window */
  def applyWindow(doc: String, s: Int, e: Int): (Array[String], Array[String], Array[String])  = {

    val target = doc.slice(s, e)
    val targetPart = tokenizer.tokenize(target.trim)
    val leftPart = if (s == 0) Array[String]()
    else tokenizer.tokenize(doc.slice(0, s).trim)

    val rightPart = if (e == doc.length) Array[String]()
    else tokenizer.tokenize(doc.slice(e, doc.length).trim)

    val (start, leftPadding) =
      if(leftPart.size >= before)
        (leftPart.size - before, Array[String]())
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


  def applyWindow(wvectors: WordEmbeddings) (doc:String, targetTerm:String, s:Int, e:Int) : Array[Double]  = {
    val tokens = doc.split(" ").filter(_!="")

    val target = Array.fill(5)(0.2)
    val nonTarget = Array.fill(5)(0.0)

    /* now start and end are indexes in the doc string */
    val start = tokens.slice(0, s).map(_.length).sum +
      tokens.slice(0, s).size // account for spaces
    val end = start + tokens.slice(s, e + 1).map(_.length).sum +
        tokens.slice(s, e + 1).size - 1 // account for spaces

    val (l, t, r) = applyWindow(doc.toLowerCase, start, end)

    l.flatMap(w => normalize(wvectors.getEmbeddings(w).map(_.toDouble))) ++
      t.flatMap(w =>  normalize(wvectors.getEmbeddings(w).map(_.toDouble)) ++ target) ++
      r.flatMap(w =>  normalize(wvectors.getEmbeddings(w).map(_.toDouble)) ++ nonTarget)
      //computeLeftDistances(l.takeRight(2), wvectors) ++ computeRightDistances(r.take(2), wvectors)
  }

  def applyWindowUdf(wvectors: WordEmbeddings, codes: Map[String, Array[Double]]) =
    udf {(doc:String, pos:mutable.WrappedArray[GenericRowWithSchema], start:Int, end:Int, targetTerm:String)  =>
      val (l, t, r) = applyWindow(doc.toLowerCase, targetTerm.toLowerCase)
      l.flatMap(w => wvectors.getEmbeddings(w)).map(_.toDouble) ++
        t.flatMap(w => wvectors.getEmbeddings(w).map(_.toDouble) ).map(_.toDouble) ++
        r.flatMap(w => wvectors.getEmbeddings(w).map(_.toDouble)  ).map(_.toDouble)
    }

  import scala.reflect.runtime.universe._
  def applyWindowUdf[T](vectors:VectorsType[T])(implicit tag: TypeTag[T]): UserDefinedFunction =
    //here 's' and 'e' are token number for start and end of target when split on " "
    udf { (doc:String, targetTerm:String, s:Int, e:Int) =>
      vectors.dense(applyWindow(wordVectors.get)(doc, targetTerm, s, e))
    }

  /*take an annotation and return vector features for it */
  def applyWindow(ann: I2b2AnnotationAndText) : Array[Double] =
     applyWindow(wordVectors.get)(ann.text, ann.target, ann.start, ann.end)



  def l2norm(xs: Array[Double]):Double = {
    import math._
    sqrt(xs.map{ x => pow(x, 2)}.sum)
  }

  def normalize(vec: Array[Double]) : Array[Double] = {
    val norm = l2norm(vec) + 1.0
    vec.map(element => element / norm)
  }

  /* experimental stuff */

  def distance(xs: Array[Float], ys: Array[Float]):Double = {
    import math.{sqrt, pow}
    sqrt((xs zip ys).map { case (x,y) => pow(y - x, 2) }.sum)
  }

  val preComplex = Map("suspicious" -> List("for"),
    "could" -> List("be"), "suggestive" -> List("of", "that"),
    "imaging" -> List("for"), "question" -> List("of"),
    "rule" -> List("out"), "evaluation" -> List("for"),
    "attributable" -> List("to"), "consistent" -> List("with"),
    "possibility" -> List("to", "of"), "in" -> List("case"), "for" -> List("presumed"), "with" ->List("possible"))

  val preSimple = Array("suggesting", "suggest", "possible", "presumed", "perhaps", "question", "investigate")

  val posComplex = Map("was" -> List("considered"))

  val posSimple = Array("vs", "or", "occurred")

  def computeLeftDistances(context: Array[String], wvectors: WordEmbeddings):
  Array[Double] = {
    //add distances to single word cues
    val single : Array[Double] = preSimple map { cue =>
      if(context.size > 0)
        distance(wvectors.getEmbeddings(cue), wvectors.getEmbeddings(context.takeRight(1).head))
      else
        -0.5 // we don't know
    }

    //add distances to complex word cues
    val complex = for ((firstToken, secondTokens) <- preComplex;
                       secondToken <- secondTokens) yield {
      if(context.size > 1)
        distance(wvectors.getEmbeddings(firstToken), wvectors.getEmbeddings(context.takeRight(2).head)) +
          distance(wvectors.getEmbeddings(secondToken), wvectors.getEmbeddings(context.takeRight(1).head))
      else
        -1.0 // we don't know
    }
    val distances:Array[Double] = single ++ complex
    normalize(distances)
  }


  def computeRightDistances(context: Array[String], wvectors: WordEmbeddings):
  Array[Double] = {


    //add distances to single word cues
    val single : Array[Double] = posSimple map { cue =>
      if(context.size > 0)
        distance(wvectors.getEmbeddings(cue), wvectors.getEmbeddings(context.head))
      else
        -0.5 // we don't know
    }

    //add distances to complex word cues
    val complex = for ((firstToken, secondTokens) <- posComplex;
                       secondToken <- secondTokens) yield {
      if(context.size > 1)
        distance(wvectors.getEmbeddings(firstToken), wvectors.getEmbeddings(context.head)) +
          distance(wvectors.getEmbeddings(secondToken), wvectors.getEmbeddings(context.tail.head))
      else
        -1.0 // we don't know
    }


    val distances = single  ++ complex
    normalize(distances)
  }

}
