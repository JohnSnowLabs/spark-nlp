package com.johnsnowlabs.nlp.annotators.assertion

import com.johnsnowlabs.nlp.annotators.assertion.logreg.{SimpleTokenizer, Windowing}
import org.scalatest.{FlatSpec, Matchers}

/**
  * Created by jose on 24/11/17.
  */
class SentenceWindowingTest extends FlatSpec with Matchers {

  trait Scope extends Windowing {
    override val before: Int = 5
    override val after: Int = 5
    override val tokenizer = new SimpleTokenizer
    override lazy val wordVectors = None
  }

  "sentences" should "be correctly padded" in new Scope {
    val doc = "the cat eats fish"
    val target = "cat"
    val result = applyWindow(doc, target)
    val expected = Array("empty_marker", "empty_marker", "empty_marker", "empty_marker",
      "the", "cat", "eats", "fish", "empty_marker", "empty_marker", "empty_marker")
    assert(expected === result.tupleToList)
  }

  "sentences" should "be correctly truncated" in new Scope {
    val doc = "it has been said that the cat eats fish while listens to the rain"
    val target = "cat"
    val expected = "has been said that the cat eats fish while listens to".split(" ")
    val result = applyWindow(doc, target)
    assert(expected === result.tupleToList)
  }

  "multi word targets" should "be correctly identified" in new Scope {
    val doc = "it has been said that the cat eats fish while listens to the rain"
    val target = "the cat"
    val expected = "it has been said that the cat eats fish while listens".split(" ")
    val result = applyWindow(doc, target)
    assert(expected === result.tupleToList)
  }

  "targets in the border" should "be correctly identified - left" in new Scope {
    val doc = "the cat eats fish while listens to the rain"
    val target = "the cat"
    val expected = ("empty_marker empty_marker empty_marker empty_marker empty_marker " +
      "the cat eats fish while listens").split(" ")
    val result = applyWindow(doc, target)
    assert(expected === result.tupleToList)
  }

  "targets in the border" should "be correctly identified - right" in new Scope {
    val doc = "it has been said that the cat"
    val target = "the cat"
    val expected = "it has been said that the cat empty_marker empty_marker empty_marker empty_marker ".split(" ")
    val result = applyWindow(doc, target)
    assert(expected === result.tupleToList)
  }

  "target occupies the whole text" should "be correctly chunked and padded" in new Scope {
    val doc = "post-operative transient ischemic attack"
    val target = "post-operative transient ischemic attack"
    val expected = ("empty_marker empty_marker empty_marker empty_marker empty_marker " +
      "post-operative transient ischemic attack empty_marker empty_marker").split(" ")

    val result = applyWindow(doc, target)
    assert(expected === result.tupleToList)
  }

  implicit class TupleOperations(t:Tuple3[Array[String], Array[String], Array[String]]) {
         def tupleToList = t._1 ++ t._2 ++ t._3
  }

}
