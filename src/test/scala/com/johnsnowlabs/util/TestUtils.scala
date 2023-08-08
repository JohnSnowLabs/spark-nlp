package com.johnsnowlabs.util

import com.johnsnowlabs.nlp.util.io.ResourceHelper
import org.scalactic.{Equality, TolerantNumerics}

import scala.io.Source

private[johnsnowlabs] object TestUtils {

  def readFile(path: String): String = {
    val stream = ResourceHelper.getResourceStream(path)
    Source.fromInputStream(stream).mkString
  }

  def assertPixels(
      values: Array[Array[Array[Float]]],
      expectedValues: Array[Array[Array[Float]]],
      error: Option[Double] = None): Unit = {
    val channels = values.length
    val width = values.head.length
    val height = values.head.head.length

    assert(expectedValues.length == channels)
    assert(expectedValues.head.length == width)
    assert(expectedValues.head.head.length == height)

    (0 until channels).foreach { channel =>
      (0 until width).foreach { w =>
        (0 until height).foreach { h =>
          val pixelVal = values(channel)(w)(h)
          val expectedPixelVal = expectedValues(channel)(w)(h)
          error match {
            case Some(err) =>
              assert(
                (pixelVal - expectedPixelVal).abs < err,
                s"Value does not match even with error: ($pixelVal, $expectedPixelVal) for $channel, $w, $h")
            case None =>
              assert(
                pixelVal == expectedPixelVal,
                s"Value does not match: ($pixelVal, $expectedPixelVal) for $channel, $w, $h")
          }

        }
      }
    }
  }

  def captureOutput(thunk: => Unit): String = {
    val stream = new java.io.ByteArrayOutputStream()
    Console.withOut(stream) {
      thunk
    }
    stream.toString
  }

  // Comparisons with tolerance, import and use ===
  implicit val tolerantFloatEq: Equality[Float] = TolerantNumerics.tolerantFloatEquality(1e-4f)
  implicit val tolerantDoubleEq: Equality[Double] = TolerantNumerics.tolerantDoubleEquality(1e-5f)

}
