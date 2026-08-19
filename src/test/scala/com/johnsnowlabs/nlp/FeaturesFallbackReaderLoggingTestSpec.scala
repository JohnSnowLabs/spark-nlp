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

package com.johnsnowlabs.nlp

import com.johnsnowlabs.nlp.annotators.SparkSessionTest
import com.johnsnowlabs.tags.FastTest
import com.johnsnowlabs.util.ConfigHelper
import org.apache.spark.SparkException
import org.apache.spark.ml.util.MLReader
import org.apache.spark.sql.SparkSession
import org.scalatest.flatspec.AnyFlatSpec

import java.io.InvalidObjectException
import java.lang.reflect.InvocationTargetException
import scala.collection.mutable.ArrayBuffer

class FeaturesFallbackReaderLoggingTestSpec extends AnyFlatSpec with SparkSessionTest {

  private class TestModel extends HasFeatures

  private class FailingReader(failure: Throwable) extends MLReader[TestModel] {
    override def load(path: String): TestModel = throw failure
  }

  private class RecordingFeaturesFallbackReader(
      baseReader: MLReader[TestModel],
      fallbackLoad: (String, SparkSession) => TestModel,
      recordedModelType: String = "TokenizerModel")
      extends FeaturesFallbackReader[TestModel](baseReader, (_, _, _) => (), fallbackLoad) {

    override protected[nlp] def modelType: String = recordedModelType

    val warnings: ArrayBuffer[(String, Option[Throwable])] = ArrayBuffer.empty

    override protected[nlp] def warn(message: String): Unit =
      warnings.append((message, None))

    override protected[nlp] def warn(message: String, throwable: Throwable): Unit =
      warnings.append((message, Some(throwable)))
  }

  private def withFallbackLogMode[T](mode: Option[String])(body: => T): T = {
    val previous = spark.conf.getOption(ConfigHelper.fallbackLoaderLogMode)
    mode match {
      case Some(value) => spark.conf.set(ConfigHelper.fallbackLoaderLogMode, value)
      case None => spark.conf.unset(ConfigHelper.fallbackLoaderLogMode)
    }

    try body
    finally {
      previous match {
        case Some(value) => spark.conf.set(ConfigHelper.fallbackLoaderLogMode, value)
        case None => spark.conf.unset(ConfigHelper.fallbackLoaderLogMode)
      }
    }
  }

  private def readerFor(
      primaryFailure: Throwable,
      fallback: (String, SparkSession) => TestModel): RecordingFeaturesFallbackReader =
    new RecordingFeaturesFallbackReader(new FailingReader(primaryFailure), fallback)
      .session(spark)

  private def isSingleLine(value: String): Boolean =
    !value.exists(character => character == '\n' || character == '\r')

  private def wrappedLambdaFailure(): Throwable = {
    val root = new IllegalArgumentException(
      "Illegal lambda\tdeserialization\nDriver stacktrace should not leak")
    val invocation = new InvocationTargetException(root)
    val invalidObject = new InvalidObjectException("Legacy serialized lambda")
    invalidObject.initCause(invocation)
    new SparkException("Spark executor failure\nDriver stacktrace", invalidObject)
  }

  behavior of "FallbackLoaderLogging"

  it should "parse supported modes case-insensitively and default to off" taggedAs FastTest in {
    assert(FallbackLoaderLogging.parseMode(None).mode == FallbackLoaderLogging.Off)
    assert(FallbackLoaderLogging.parseMode(Some("off")).mode == FallbackLoaderLogging.Off)
    assert(
      FallbackLoaderLogging.parseMode(Some(" SuMmArY ")).mode == FallbackLoaderLogging.Summary)
    assert(FallbackLoaderLogging.parseMode(Some("FULL")).mode == FallbackLoaderLogging.Full)
  }

  it should "treat an invalid mode as off while preserving the invalid value" taggedAs FastTest in {
    val parsed = FallbackLoaderLogging.parseMode(Some("verbose"))

    assert(parsed.mode == FallbackLoaderLogging.Off)
    assert(parsed.invalidValue.contains("verbose"))
  }

  it should "format the deepest meaningful cause as one bounded line" taggedAs FastTest in {
    val message = FallbackLoaderLogging.summary("TokenizerModel", wrappedLambdaFailure())

    assert(isSingleLine(message))
    assert(message.contains("Spark NLP fallback loader activated for TokenizerModel"))
    assert(message.contains("IllegalArgumentException: Illegal lambda deserialization"))
    assert(!message.contains("Spark executor failure"))
    assert(!message.contains("Driver stacktrace"))
    assert(!message.contains("/tmp/legacy/model"))
    assert(message.contains(s"${ConfigHelper.fallbackLoaderLogMode}=full"))
  }

  it should "bound only dynamic exception text to 200 characters" taggedAs FastTest in {
    val dynamicText = "x" * 250
    val message =
      FallbackLoaderLogging.summary("TokenizerModel", new IllegalArgumentException(dynamicText))
    val renderedDynamicText = message
      .stripPrefix(
        "Spark NLP fallback loader activated for TokenizerModel. Cause: IllegalArgumentException: ")
      .takeWhile(_ != '.')

    assert(renderedDynamicText.length == FallbackLoaderLogging.MaxCauseMessageLength)
    assert(
      message.endsWith(
        s"Set ${ConfigHelper.fallbackLoaderLogMode}=full for the complete stack trace."))
  }

  behavior of "FeaturesFallbackReader"

  it should "run fallback once without warnings when the mode is absent" taggedAs FastTest in {
    var fallbackInvocations = 0
    val fallbackResult = new TestModel
    val reader = readerFor(
      wrappedLambdaFailure(),
      (_, _) => {
        fallbackInvocations += 1
        fallbackResult
      })

    val result = withFallbackLogMode(None)(reader.load("/tmp/legacy/model"))

    assert(result eq fallbackResult)
    assert(fallbackInvocations == 1)
    assert(reader.warnings.isEmpty)
  }

  it should "run fallback once without warnings when the mode is off" taggedAs FastTest in {
    var fallbackInvocations = 0
    val fallbackResult = new TestModel
    val reader = readerFor(
      wrappedLambdaFailure(),
      (_, _) => {
        fallbackInvocations += 1
        fallbackResult
      })

    val result = withFallbackLogMode(Some("OFF"))(reader.load("/tmp/legacy/model"))

    assert(result eq fallbackResult)
    assert(fallbackInvocations == 1)
    assert(reader.warnings.isEmpty)
  }

  it should "emit one summary warning without attaching the exception" taggedAs FastTest in {
    val primaryFailure = wrappedLambdaFailure()
    val fallbackResult = new TestModel
    val reader = readerFor(primaryFailure, (_, _) => fallbackResult)

    val result = withFallbackLogMode(Some("summary"))(reader.load("/tmp/legacy/model"))

    assert(result eq fallbackResult)
    assert(reader.warnings.size == 1)
    assert(reader.warnings.head._2.isEmpty)
    assert(isSingleLine(reader.warnings.head._1))
    assert(!reader.warnings.head._1.contains("/tmp/legacy/model"))
    assert(!reader.warnings.head._1.contains("Driver stacktrace"))
  }

  it should "emit the same summary and attach the original exception in full mode" taggedAs FastTest in {
    val primaryFailure = wrappedLambdaFailure()
    val fallbackResult = new TestModel
    val reader = readerFor(primaryFailure, (_, _) => fallbackResult)

    withFallbackLogMode(Some("full"))(reader.load("/tmp/legacy/model"))

    assert(reader.warnings.size == 1)
    assert(
      reader.warnings.head._1 == FallbackLoaderLogging.summary("TokenizerModel", primaryFailure))
    assert(reader.warnings.head._2.exists(_ eq primaryFailure))
  }

  it should "warn about an invalid mode and still run fallback once" taggedAs FastTest in {
    var fallbackInvocations = 0
    val fallbackResult = new TestModel
    val reader = readerFor(
      wrappedLambdaFailure(),
      (_, _) => {
        fallbackInvocations += 1
        fallbackResult
      })

    val result = withFallbackLogMode(Some("verbose\nmode"))(reader.load("/tmp/legacy/model"))

    assert(result eq fallbackResult)
    assert(fallbackInvocations == 1)
    assert(reader.warnings.size == 1)
    assert(reader.warnings.head._2.isEmpty)
    assert(isSingleLine(reader.warnings.head._1))
    assert(reader.warnings.head._1.contains("verbose mode"))
    assert(reader.warnings.head._1.contains("off, summary, and full"))
    assert(!reader.warnings.head._1.contains("fallback loader activated"))
  }

  it should "propagate fallback failures unchanged" taggedAs FastTest in {
    val fallbackFailure = new IllegalStateException("fallback failed")
    val reader = readerFor(wrappedLambdaFailure(), (_, _) => throw fallbackFailure)

    val thrown = withFallbackLogMode(Some("summary")) {
      intercept[IllegalStateException](reader.load("/tmp/legacy/model"))
    }

    assert(thrown eq fallbackFailure)
    assert(reader.warnings.size == 1)
  }

  it should "run fallback even if warning emission fails" taggedAs FastTest in {
    var fallbackInvocations = 0
    val fallbackResult = new TestModel
    val reader = new RecordingFeaturesFallbackReader(
      new FailingReader(wrappedLambdaFailure()),
      (_, _) => {
        fallbackInvocations += 1
        fallbackResult
      }) {
      override protected[nlp] def warn(message: String): Unit =
        throw new IllegalStateException("logger failed")
    }.session(spark)

    val result = withFallbackLogMode(Some("summary"))(reader.load("/tmp/legacy/model"))

    assert(result eq fallbackResult)
    assert(fallbackInvocations == 1)
  }
}
