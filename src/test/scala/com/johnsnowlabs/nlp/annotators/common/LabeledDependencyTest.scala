package com.johnsnowlabs.nlp.annotators.common

import com.johnsnowlabs.nlp.Annotation
import com.johnsnowlabs.nlp.AnnotatorType.{DEPENDENCY, LABELED_DEPENDENCY, TOKEN}
import com.johnsnowlabs.nlp.annotators.common.LabeledDependency.DependencyInfo
import org.scalatest.FlatSpec

class LabeledDependencyTest extends FlatSpec {

  "LabeledDependencyTest" should "unpack Dependency and Typed Dependency Parser annotators" in {

    val mockAnnotations = getMockAnnotations()
    val expectedOutput = Seq(
      DependencyInfo(0, 5, "United", 7, 14, 2, "canceled", "nsubj"),
      DependencyInfo(7, 14, "canceled", -1, -1, 0,"*ROOT*", "*root*"),
      DependencyInfo(16, 18, "the", 28, 34, 5,"flights", "det"),
      DependencyInfo(20, 26, "morning", 28, 34, 5, "flights", "compound"),
      DependencyInfo(28, 34, "flights", 7, 14, 2, "canceled", "obj"),
      DependencyInfo(36, 37, "to", 39, 45, 7, "Houston", "case"),
      DependencyInfo(39, 45, "Houston", 28, 34, 5, "flights", "nmod")
    )

    val actualOutput = LabeledDependency.unpackHeadAndRelation(mockAnnotations)

    assert(expectedOutput == actualOutput)
  }

  it should "raise error when annotations length are not the same" in {

    val mockAnnotations = getMockAnnotations(missingRecord = true)

    assertThrows[IndexOutOfBoundsException] {
      LabeledDependency.unpackHeadAndRelation(mockAnnotations)
    }
  }

  private def getMockAnnotations(missingRecord: Boolean = false): Seq[Annotation] = {

    val mockTokenizer = Seq(
      Annotation(TOKEN, 0, 5, "United", Map("sentence" -> "0")),
      Annotation(TOKEN, 7, 14, "canceled", Map("sentence" -> "0")),
      Annotation(TOKEN, 16, 18, "the", Map("sentence" -> "0")),
      Annotation(TOKEN, 20, 26, "morning", Map("sentence" -> "0")),
      Annotation(TOKEN, 28, 34, "flights", Map("sentence" -> "0")),
      Annotation(TOKEN, 36, 37, "to", Map("sentence" -> "0")),
      Annotation(TOKEN, 39, 45, "Houston", Map("sentence" -> "0"))
    )

    val mockDependencyParser = Seq(
      Annotation(DEPENDENCY, 0, 5, "canceled", Map("head" -> "2", "head.begin" -> "7", "head.end" -> "14")),
      Annotation(DEPENDENCY, 7, 14, "ROOT", Map("head" -> "0", "head.begin" -> "-1", "head.end" -> "-1")),
      Annotation(DEPENDENCY, 16, 18, "flights", Map("head" -> "5", "head.begin" -> "28", "head.end" -> "34")),
      Annotation(DEPENDENCY, 20, 26, "flights", Map("head" -> "5", "head.begin" -> "28", "head.end" -> "34")),
      Annotation(DEPENDENCY, 28, 34, "canceled", Map("head" -> "2", "head.begin" -> "7", "head.end" -> "14")),
      Annotation(DEPENDENCY, 36, 37, "Houston", Map("head" -> "7", "head.begin" -> "39", "head.end" -> "45")),
      Annotation(DEPENDENCY, 39, 45, "flights", Map("head" -> "5", "head.begin" -> "28", "head.end" -> "34"))
    )

    val mockTypedDependencyParser = {
      Seq(
        Annotation(LABELED_DEPENDENCY, 0, 5, "nsubj", Map()),
        Annotation(LABELED_DEPENDENCY, 7, 14, "root", Map()),
        Annotation(LABELED_DEPENDENCY, 16, 18, "det", Map()),
        Annotation(LABELED_DEPENDENCY, 20, 26, "compound", Map()),
        Annotation(LABELED_DEPENDENCY, 28, 34, "obj", Map()),
        Annotation(LABELED_DEPENDENCY, 36, 37, "case", Map()),
        Annotation(LABELED_DEPENDENCY, 39, 45, "nmod", Map())
      )
    }

    if (missingRecord) {
      mockTokenizer ++ mockDependencyParser ++ mockTypedDependencyParser.slice(0, mockTypedDependencyParser.length - 2 )
    } else {
      mockTokenizer ++ mockDependencyParser ++ mockTypedDependencyParser
    }
  }

}
