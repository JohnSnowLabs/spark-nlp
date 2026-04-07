/*
 * Copyright 2017-2025 John Snow Labs
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
package com.johnsnowlabs.reader.util

import com.johnsnowlabs.tags.FastTest
import com.johnsnowlabs.util.LocalHttpTestServer
import org.scalatest.flatspec.AnyFlatSpec

import java.io.IOException
import java.util.Base64

class ImageParserTest extends AnyFlatSpec {

  private val tinyPngBase64 =
    "iVBORw0KGgoAAAANSUhEUgAAAAUA" +
      "AAAFCAYAAACNbyblAAAAHElEQVQI12P4//8/w38GIAXDIBKE0DHxgljNBAAO9TXL0Y4OHwAAAABJRU5ErkJggg=="

  "ImageHelper" should "convert image to base64" taggedAs FastTest in {
    val base64 =
      "iVBORw0KGgoAAAANSUhEUgAAAAUA\n  AAAFCAYAAACNbyblAAAAHElEQVQI12P4\n  //8/w38GIAXDIBKE0DHxgljNBAAO9TXL0Y4OHwAAAABJRU5ErkJggg=="

    val decodedImage = ImageParser.decodeBase64(base64)

    assert(decodedImage.isDefined)
    assert(decodedImage.get.getHeight > 0)
    assert(decodedImage.get.getWidth > 0)
    assert(decodedImage.get.getType > 0)
  }

  it should "fetch image from a valid HTTP URL" taggedAs FastTest in {
    val pngBytes = Base64.getDecoder.decode(tinyPngBase64)
    val server = LocalHttpTestServer.start(
      Map("/image.png" -> LocalHttpTestServer.Response(200, pngBytes, contentType = "image/png")))

    try {
      val resultImage = ImageParser.fetchFromUrl(server.url("/image.png"))

      assert(resultImage.isDefined)
      assert(resultImage.get.getHeight > 0)
      assert(resultImage.get.getWidth > 0)
      assert(resultImage.get.getType > 0)
    } finally {
      server.close()
    }
  }

  it should "throw for a non-success HTTP response" taggedAs FastTest in {
    val server = LocalHttpTestServer.start(
      Map("/missing.png" -> LocalHttpTestServer
        .Response(404, "Not found".getBytes("UTF-8"), contentType = "text/plain")))

    try {
      assertThrows[IOException] {
        ImageParser.fetchFromUrl(server.url("/missing.png"))
      }
    } finally {
      server.close()
    }
  }

  it should "throw when the URL is unreachable" taggedAs FastTest in {
    assertThrows[IOException] {
      ImageParser.fetchFromUrl("http://127.0.0.1:1/unreachable.png")
    }
  }

}
