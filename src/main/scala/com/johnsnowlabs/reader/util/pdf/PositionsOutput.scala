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
package com.johnsnowlabs.reader.util.pdf

import com.johnsnowlabs.nlp.IAnnotation
import com.johnsnowlabs.reader.util.pdf.schema.MappingMatrix

import java.util

case class PositionsOutput(mappings: Array[MappingMatrix]) extends IAnnotation {
  override def annotatorType: String = "page_matrix"

  def toPython = {
    val javaMappings = new java.util.ArrayList[util.HashMap[String, Any]]()
    mappings.foreach { case MappingMatrix(c, x, y, width, height, fontSize, source) =>
      val javaMapping = new util.HashMap[String, Any]()
      javaMapping.put("c", c)
      javaMapping.put("x", x)
      javaMapping.put("y", y)

      javaMapping.put("width", width)
      javaMapping.put("height", height)

      javaMapping.put("fontSize", fontSize)
      javaMapping.put("source", source)

      javaMappings.add(javaMapping)

    }
    javaMappings
  }
}
