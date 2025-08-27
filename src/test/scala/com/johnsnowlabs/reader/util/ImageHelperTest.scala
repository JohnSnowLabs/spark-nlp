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

import org.scalatest.flatspec.AnyFlatSpec

class ImageHelperTest extends AnyFlatSpec {

  "ImageHelper" should "convert image to base64" in {
    val base64 =
      "iVBORw0KGgoAAAANSUhEUgAAAAUA\n  AAAFCAYAAACNbyblAAAAHElEQVQI12P4\n  //8/w38GIAXDIBKE0DHxgljNBAAO9TXL0Y4OHwAAAABJRU5ErkJggg=="

    val decodedImage = ImageHelper.decodeBase64(base64)

    assert(decodedImage.isDefined)
    assert(decodedImage.get.getHeight > 0)
    assert(decodedImage.get.getWidth > 0)
    assert(decodedImage.get.getType > 0)
  }

  it should "fail to convert invalid base64" in {
    val url =
      "https://upload.wikimedia.org/wikipedia/commons/thumb/a/a7/React-icon.svg/1024px-React-icon.svg.png"

    val resultImage = ImageHelper.fetchFromUrl(url)

    assert(resultImage.isDefined)
    assert(resultImage.get.getHeight > 0)
    assert(resultImage.get.getWidth > 0)
    assert(resultImage.get.getType > 0)
  }

}
