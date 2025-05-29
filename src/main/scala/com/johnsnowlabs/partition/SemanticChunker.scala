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
package com.johnsnowlabs.partition

import com.johnsnowlabs.partition.BasicChunker.chunkBasic
import com.johnsnowlabs.reader.HTMLElement
import com.johnsnowlabs.reader.util.PartitionOptions.{getDefaultInt, getDefaultString}
import org.apache.spark.sql.Row
import org.apache.spark.sql.expressions.UserDefinedFunction
import org.apache.spark.sql.functions.udf

import scala.collection.mutable

class SemanticChunker(chunkerOptions: Map[String, String]) extends Serializable {

  def chunkUDF(): UserDefinedFunction = {
    udf((elements: Seq[Row]) => {
      val htmlElements = elements.map { row =>
        val elementType = row.getAs[String]("elementType")
        val content = row.getAs[String]("content")
        val metadata = row.getAs[Map[String, String]]("metadata")
        HTMLElement(elementType, content, mutable.Map.empty ++ metadata)
      }.toList

      val chunks = getChunkerStrategy match {
        case "basic" => chunkBasic(htmlElements, getMaxCharacters, getNewAfterNChars, getOverlap)
        case _ =>
          throw new IllegalArgumentException(s"Unknown chunker strategy: $getChunkerStrategy")
      }

      chunks.flatMap(_.elements)
    })
  }

  private def getMaxCharacters: Int = {
    getDefaultInt(chunkerOptions, Seq("maxCharacters", "max_characters"), default = 500)
  }

  private def getNewAfterNChars: Int = {
    getDefaultInt(chunkerOptions, Seq("newAfterNChars", "new_after_n_chars"), default = -1)
  }

  private def getOverlap: Int = {
    getDefaultInt(chunkerOptions, Seq("overlap", "overlap"), default = 0)
  }

  private def getChunkerStrategy: String = {
    getDefaultString(
      chunkerOptions,
      Seq("chunkingStrategy", "chunking_strategy"),
      default = "none")
  }

}
