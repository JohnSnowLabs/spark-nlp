package com.johnsnowlabs.tags

import org.scalatest.{Args, Filter, Reporter, Tag}
import org.scalatest.events.{Event, TestSucceeded}
import org.scalatest.funsuite.AnyFunSuite

import scala.collection.mutable.ArrayBuffer

/** Cheap real ScalaTest cases: no Spark session, model or network access. */
class TaxonomyControlSpec extends AnyFunSuite {
  val executed: ArrayBuffer[String] = ArrayBuffer.empty[String]
  private val tensorflow = new Tag("com.johnsnowlabs.tags.TensorFlow")
  private val embeddings = new Tag("com.johnsnowlabs.tags.Embeddings")
  private val text = new Tag("com.johnsnowlabs.tags.Text")
  private val heavy = new Tag("com.johnsnowlabs.tags.ResourceIntensiveTest")

  test("untagged control") { executed += "untagged control" }
  test("fast control", FastTest) { executed += "fast control" }
  test("light embeddings control", tensorflow, embeddings, text) {
    executed += "light embeddings control"
  }
  test("heavy embeddings control", SlowTest, tensorflow, embeddings, text, heavy) {
    executed += "heavy embeddings control"
  }
}

class TaxonomySelectionSpec extends AnyFunSuite {
  private val silent = new Reporter { override def apply(event: Event): Unit = () }
  private def selected(expression: String): Set[String] =
    TaxonomyRunner.select(new TaxonomyControlSpec, expression).toSet

  test("discover real testNames and tags without requiring taxonomy on every case") {
    val suite = new TaxonomyControlSpec
    assert(TaxonomyRunner.select(suite, "NOT SlowTest").toSet == Set(
      "untagged control", "fast control", "light embeddings control"))
    assert(suite.executed.isEmpty)
  }

  test("intersection and exclusion select only the light embeddings case") {
    assert(selected("TensorFlow AND Embeddings AND NOT ResourceIntensiveTest") ==
      Set("light embeddings control"))
  }

  test("OR overlaps execute exact test IDs once") {
    val suite = new TaxonomyControlSpec
    val names = TaxonomyRunner.select(suite, "TensorFlow OR Embeddings")
    val successes = ArrayBuffer.empty[String]
    val reporter = new Reporter {
      override def apply(event: Event): Unit = event match {
        case e: TestSucceeded => successes += e.testName
        case _ => ()
      }
    }
    TaxonomyRunner.execute(suite, names, reporter)
    assert(suite.executed.sorted.toVector == names.sorted)
    assert(successes.sorted.toVector == names.sorted)
    assert(names.toSet == Set("light embeddings control", "heavy embeddings control"))
  }

  test("NOT binds before AND which binds before OR and parentheses override precedence") {
    assert(selected("FastTest OR TensorFlow AND NOT ResourceIntensiveTest") ==
      Set("fast control", "light embeddings control"))
    assert(selected("(FastTest OR TensorFlow) AND ResourceIntensiveTest") ==
      Set("heavy embeddings control"))
    assert(selected("NOT (SlowTest OR FastTest)") ==
      Set("untagged control", "light embeddings control"))
    assert(selected("NOT NOT ResourceIntensiveTest") == Set("heavy embeddings control"))
  }

  test("unknown names fail even in a Boolean branch that would short circuit") {
    Seq("Tensorflow", "TensorFlow OR Typo", "FastTest AND Typo", "NOT Typo").foreach { expression =>
      val error = intercept[IllegalArgumentException] {
        TaxonomyRunner.select(new TaxonomyControlSpec, expression)
      }
      assert(error.getMessage.contains("Unknown taxonomy name"))
    }
  }

  test("malformed Boolean expressions fail closed") {
    Seq("", " ", "TensorFlow AND", "AND Text", "()", "(Text", "Text)",
      "Text Embeddings", "Text & Embeddings", "Text OR OR Text").foreach { expression =>
      intercept[IllegalArgumentException] {
        TaxonomyRunner.select(new TaxonomyControlSpec, expression)
      }
    }
  }

  test("empty selection requires explicit opt in") {
    val suite = new TaxonomyControlSpec
    intercept[IllegalArgumentException] {
      TaxonomyRunner.select(suite, "TensorFlow AND NOT Embeddings")
    }
    assert(TaxonomyRunner.select(suite, "TensorFlow AND NOT Embeddings", allowEmpty = true).isEmpty)
    assert(suite.executed.isEmpty)
  }

  test("ordinary non slow filtering still includes the untagged and fast controls") {
    val suite = new TaxonomyControlSpec
    val status = suite.run(None, Args(silent,
      filter = Filter(tagsToExclude = Set(SlowTest.name))))
    status.waitUntilCompleted()
    assert(status.succeeds())
    assert(suite.executed.toSet == Set(
      "untagged control", "fast control", "light embeddings control"))
  }

  test("selected failures and cancellations cannot report success") {
    val failing = new AnyFunSuite { test("failed") { fail("expected witness failure") } }
    intercept[IllegalStateException] {
      TaxonomyRunner.execute(failing, Vector("failed"), silent)
    }
    val canceled = new AnyFunSuite { test("canceled") { cancel("expected witness cancellation") } }
    intercept[IllegalStateException] {
      TaxonomyRunner.execute(canceled, Vector("canceled"), silent)
    }
  }

  test("selected ignored cases cannot silently pass and nested suites do not run") {
    val ignored = new AnyFunSuite { ignore("ignored") { fail("must not run") } }
    intercept[IllegalStateException] {
      TaxonomyRunner.execute(ignored, Vector("ignored"), silent)
    }
    val nested = new AnyFunSuite {
      test("direct") { succeed }
      override val nestedSuites = Vector(new AnyFunSuite {
        test("nested") { fail("nested suites are outside the explicit suite scope") }
      })
    }
    TaxonomyRunner.execute(nested, Vector("direct"), silent)
  }

  test("python-style lowercase expressions select the same IDs") {
    assert(selected("tensorflow and embeddings and not resource_intensive") ==
      Set("light embeddings control"))
    assert(selected("TensorFlow AND Embeddings AND NOT ResourceIntensiveTest") ==
      Set("light embeddings control"))
  }

  test("ALBERT pilot has exactly one classified case and is not resource-intensive") {
    val suite = new com.johnsnowlabs.nlp.embeddings.AlbertEmbeddingsTestSpec
    val name = "AlbertEmbeddings should be aligned with custom tokens from Tokenizer"
    assert(TaxonomyRunner.select(suite, "onnx and embeddings and text") == Vector(name))
    assert(suite.tags(name).contains(SlowTest.name))
    assert(suite.tags(name).contains(ONNX.name))
    assert(!suite.tags(name).contains(TensorFlow.name))
    assert(!suite.tags(name).contains(ResourceIntensiveTest.name))
    assert(TaxonomyRunner.select(suite,
      "onnx and embeddings and text and not resource_intensive") == Vector(name))
    assert(suite.testNames.size == 4)
  }
}
