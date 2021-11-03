/*
 * Copyright 2017-2021 John Snow Labs
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

package com.johnsnowlabs.nlp.annotators.common

import java.util.regex.Pattern

trait PreprocessingParser {
  def separate(token:String): String
}


class SuffixedToken(suffixes:Array[String]) extends PreprocessingParser {

  def belongs(token: String): Option[String] =
    suffixes.find(token.endsWith)


  override def separate(token:String): String = {
    belongs(token).map { suffix  =>
      s"""${separate(token.dropRight(suffix.length))} $suffix"""
    }.getOrElse(token)
  }

}

object SuffixedToken {
  def apply(suffixes:Array[String]) = new SuffixedToken(suffixes)
}


class PrefixedToken(prefixes:Array[String]) extends PreprocessingParser {

  private def parse(token:String)  =
    (token.head.toString, token.tail)

  def belongs(token: String): Option[String] =
      prefixes.find(token.startsWith)


  override def separate(token:String): String = {
    belongs(token).map { prefix =>
      s"""$prefix ${separate(token.drop(prefix.length))}"""
    }.getOrElse(token)
  }
}

object PrefixedToken {
  def apply(prefixes:Array[String]) = new PrefixedToken(prefixes)
}


class InfixToken(tokens:Array[String]) extends PreprocessingParser {

  private def parse(token:String)  =
    (token.head.toString, token.tail)

  def belongs(token: String): Boolean = {
    if(token.length > 2) {
      val insideChunk = token.tail.dropRight(1)
      tokens.exists(insideChunk.contains)
    }else{
      false
    }
  }

  override def separate(token:String): String = {
    var result = token
    if (belongs(token)) {
      tokens.foreach{ t =>
        val quotedInfix = Pattern.quote(t)
        result = result.replaceAll(quotedInfix, s" $t ")
      }
    }
    result
  }
}

object InfixToken {
  def apply(infixes:Array[String]) = new InfixToken(infixes)
}