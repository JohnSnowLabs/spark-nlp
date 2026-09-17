package com.johnsnowlabs.tags

import org.scalatest.{Args, DynaTags, Filter, Reporter, Suite}
import org.scalatest.events._

import scala.collection.mutable.ArrayBuffer

/** Opt-in Boolean selection over one explicitly named suite's real ScalaTest metadata.
  *
  * Expressions use the short, case-sensitive names in TestTaxonomy. NOT binds before AND,
  * which binds before OR. No taxonomy requirement is imposed on unclassified tests.
  */
object TaxonomyRunner {
  private type Predicate = Set[String] => Boolean

  // Parse every operand before evaluation, including branches that would short circuit.
  private def parse(expression: String): Predicate = {
    val tokens = "[A-Za-z][A-Za-z0-9_]*|[()]|\\S".r.findAllIn(expression).toVector
    var position = 0
    def take(token: String, ignoreCase: Boolean = false): Boolean = {
      val current = tokens.lift(position)
      val matched =
        if (ignoreCase) current.exists(_.equalsIgnoreCase(token)) else current.contains(token)
      if (matched) {
        position += 1
        true
      } else false
    }
    def atom(): Predicate = {
      if (take("NOT", ignoreCase = true)) {
        val child = atom()
        tags => !child(tags)
      } else if (take("(")) {
        val child = disjunction()
        require(take(")"), "Expected ')' in taxonomy expression")
        child
      } else {
        require(position < tokens.size, "Expected taxonomy name or '('")
        val name = tokens(position)
        position += 1
        val tag = TestTaxonomy.names.getOrElse(name,
          throw new IllegalArgumentException(s"Unknown taxonomy name: $name. Known names: " +
            TestTaxonomy.names.keys.toVector.sorted.mkString(", ")))
        tags => tags.contains(tag.name)
      }
    }
    def conjunction(): Predicate = {
      var result = atom()
      while (take("AND", ignoreCase = true)) {
        val left = result
        val right = atom()
        result = tags => left(tags) && right(tags)
      }
      result
    }
    def disjunction(): Predicate = {
      var result = conjunction()
      while (take("OR", ignoreCase = true)) {
        val left = result
        val right = conjunction()
        result = tags => left(tags) || right(tags)
      }
      result
    }
    val result = disjunction()
    require(position == tokens.size,
      s"Unexpected token in taxonomy expression: ${tokens.lift(position).getOrElse("")}")
    result
  }

  def select(suite: Suite, expression: String, allowEmpty: Boolean = false): Vector[String] =
    select(suite, parse(expression), allowEmpty)

  private def select(suite: Suite, predicate: Predicate, allowEmpty: Boolean): Vector[String] = {
    val selected = suite.testNames.toVector.sorted.filter { name =>
      predicate(suite.tags.getOrElse(name, Set.empty))
    }
    require(selected.nonEmpty || allowEmpty,
      s"Taxonomy expression selected no tests in ${suite.suiteId}; use --allow-empty intentionally")
    selected
  }

  /** Use an exact-name dynamic tag, not ScalaTest's OR-only tag CLI or substring selectors.
    * Keep suite fixtures and status handling, and prohibit traversal into nested suites.
    */
  def execute(suite: Suite, names: Vector[String], reporter: Reporter): Unit = {
    require(names.distinct == names, "Duplicate selected test IDs")
    require(names.forall(suite.testNames.contains), "Selected test ID absent from suite.testNames")
    if (names.nonEmpty) {
      val marker = "com.johnsnowlabs.tags.TaxonomyRunner.Selected"
      val succeeded = ArrayBuffer.empty[String]
      val started = ArrayBuffer.empty[String]
      val recording = new Reporter {
        override def apply(event: Event): Unit = synchronized {
          event match {
            case e: TestStarting => started += e.testName
            case e: TestSucceeded => succeeded += e.testName
            case _ => ()
          }
          reporter(event)
        }
      }
      val filter = Filter(
        tagsToInclude = Some(Set(marker)),
        tagsToExclude = Set.empty,
        excludeNestedSuites = true,
        dynaTags = DynaTags(Map.empty,
          Map(suite.suiteId -> names.map(_ -> Set(marker)).toMap)))
      val status = suite.run(None, Args(recording, filter = filter))
      status.waitUntilCompleted()
      if (!status.succeeds() || started.sorted.toVector != names.sorted ||
          succeeded.sorted.toVector != names.sorted) {
        throw new IllegalStateException(
          s"Taxonomy execution failed: selected=${names.size}, started=${started.size}, " +
            s"succeeded=${succeeded.size}. Every selected test must succeed exactly once.")
      }
    }
  }

  private val consoleReporter = new Reporter {
    override def apply(event: Event): Unit = event match {
      case e: TestStarting => println(s"TAXONOMY START ${e.suiteId} :: ${e.testName}")
      case e: TestSucceeded => println(s"TAXONOMY PASS ${e.suiteId} :: ${e.testName}")
      case e: TestFailed =>
        Console.err.println(s"TAXONOMY FAIL ${e.suiteId} :: ${e.testName}: ${e.message}")
        e.throwable.foreach(_.printStackTrace())
      case e: TestCanceled => Console.err.println(s"TAXONOMY CANCELED ${e.testName}: ${e.message}")
      case e: TestIgnored => Console.err.println(s"TAXONOMY IGNORED ${e.testName}")
      case e: TestPending => Console.err.println(s"TAXONOMY PENDING ${e.testName}")
      case e: SuiteAborted =>
        Console.err.println(s"TAXONOMY ABORT ${e.suiteId}: ${e.message}")
        e.throwable.foreach(_.printStackTrace())
      case _ => ()
    }
  }

  def main(args: Array[String]): Unit = {
    require(args.length >= 2,
      "Usage: taxonomyTest <fully.qualified.Suite> \"Boolean expression\" [--discover] [--allow-empty]")
    val suiteName = args(0)
    require(suiteName.matches("[A-Za-z_$][A-Za-z0-9_$]*(\\.[A-Za-z_$][A-Za-z0-9_$]*)+"),
      "Specify exactly one fully qualified suite class; wildcards are not supported")
    val options = args.drop(2).toVector
    require(options.distinct.size == options.size &&
      options.forall(Set("--discover", "--allow-empty")),
      "Expected a quoted expression followed only by --discover and/or --allow-empty")
    val predicate = parse(args(1)) // Validate unknown names before constructing the suite.
    val suite = Class.forName(suiteName).asSubclass(classOf[Suite]).getConstructor().newInstance()
    val names = select(suite, predicate, options.contains("--allow-empty"))
    println(s"TAXONOMY suite=$suiteName discovered=${suite.testNames.size} selected=${names.size}")
    names.foreach { name =>
      println(s"TAXONOMY SELECT $suiteName :: $name tags=" +
        suite.tags.getOrElse(name, Set.empty).toVector.sorted.mkString(","))
    }
    if (!options.contains("--discover")) {
      execute(suite, names, consoleReporter)
      println(s"TAXONOMY RESULT succeeded=${names.size} failed=0")
    }
  }
}
