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
package com.johnsnowlabs.nlp.annotators.er

import com.johnsnowlabs.nlp.Annotation
import com.johnsnowlabs.nlp.AnnotatorType.CHUNK
import com.johnsnowlabs.nlp.annotators.common.Sentence

import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer

/** Aho-Corasick Algorithm: https://dl.acm.org/doi/10.1145/360825.360855 A simple, efficient
  * algorithm to locate all occurrences of any of a finite number of keywords in a string of text.
  * The algorithm consists of constructing a finite state pattern matching machine from the
  * keywords and then using the pattern matching machine to process the text string in a single
  * pass. The complexity of constructing a pattern matching machine and searching the text is
  * linear to the total length of given patterns and the length of a text, respectively.
  */
class AhoCorasickAutomaton(
    var alphabet: String,
    patterns: Array[EntityPattern],
    caseSensitive: Boolean = false)
    extends Serializable {

  alphabet = if (alphabet.contains(" ")) alphabet else alphabet + " "

  private val flattenEntityPatterns: Array[FlattenEntityPattern] = patterns.flatMap {
    entityPattern =>
      entityPattern.patterns.map { pattern =>
        val keyword = if (caseSensitive) pattern else pattern.toLowerCase
        FlattenEntityPattern(entityPattern.label, keyword, entityPattern.id)
      }
  }

  private val MAX_STATES: Int = flattenEntityPatterns.map(value => value.keyword.length).sum + 1
  private val MAX_EDGES: Int = alphabet.length

  private val edges: Map[Char, Int] = alphabet.toCharArray.zipWithIndex.map {
    case (char, index) => (char, index)
  }.toMap
  private val output: mutable.Map[Int, Int] = mutable.Map(0 -> 0)
  private val failureLink: mutable.Map[Int, Int] = mutable.Map(0 -> -1)

  /** Transition Table aka Goto function: Using a 2D Array to represent a Trie */
  private val transitionTable: Array[Array[Int]] =
    Array.fill(MAX_STATES)(Array.fill(MAX_EDGES)(-1))

  /** First step of Aho-Corasick algorithm: Build a Finite State Automaton as a keyword trie in
    * which the nodes represent the state and the edges between nodes are labeled by characters
    * that cause the transitions between nodes. The trie is an efficient implementation of a
    * dictionary of strings. It is also easy to implement a state machine as a lookup table where
    * the rows represent states and the columns represent input character
    */
  def buildMatchingMachine(): Unit = {
    buildGoToGraph()
    buildFailureLink()
  }

  private def buildGoToGraph(): Unit = {

    var state = 1

    flattenEntityPatterns.zipWithIndex.foreach { case (flattenEntityPattern, keywordIndex) =>
      val keyword = flattenEntityPattern.keyword
      var currentState = 0

      keyword.foreach { char =>
        val currentChar = if (caseSensitive) char else char.toLower
        val edgeIndex = edges.getOrElse(
          currentChar,
          throw new UnsupportedOperationException(
            s"Char $currentChar not found on alphabet. Please check alphabet"))
        // Allocate a new node (create a new state) if a node for character doesn't exist.
        if (transitionTable(currentState)(edgeIndex) == -1) {
          transitionTable(currentState)(edgeIndex) = state
          state = state + 1
        }

        currentState = transitionTable(currentState)(edgeIndex)
      }

      addCurrentKeywordInOutput(currentState, keywordIndex)
    }

    addGoToEdgesFromZeroStatesToRoot()
  }

  private def addCurrentKeywordInOutput(state: Int, keywordIndex: Int): Unit = {
    val bitwiseOrResult = output.getOrElse(state, 0) | 1 << keywordIndex
    if (bitwiseOrResult > 0) {
      output(state) = bitwiseOrResult
    }
  }

  private def addGoToEdgesFromZeroStatesToRoot(): Unit = {
    for (charIndex <- 0 until MAX_EDGES) {
      if (transitionTable(0)(charIndex) == -1) {
        transitionTable(0)(charIndex) = 0
      }
    }
  }

  /** Failure function is computed in BFS order */
  private def buildFailureLink(): Unit = {

    val queue: mutable.Queue[Int] = mutable.Queue()

    for (edgeIndex <- 0 until MAX_EDGES) {
      if (transitionTable(0)(edgeIndex) != 0) {
        val nodeDepthOne = transitionTable(0)(edgeIndex)
        failureLink(nodeDepthOne) = 0

        queue.enqueue(nodeDepthOne)
      }
    }

    while (queue.nonEmpty) {
      val state = queue.dequeue
      // Find failure function for all those characters for which goto function is not defined.
      for (edgeIndex <- 0 until MAX_EDGES) {

        if (transitionTable(state)(edgeIndex) != -1) {
          val failure = findFailureState(state, edgeIndex)
          val nextNode = transitionTable(state)(edgeIndex)
          failureLink(nextNode) = failure

          mergeSubstringToOutput(nextNode, failure)

          queue.enqueue(nextNode)
        }
      }
    }
  }

  private def mergeSubstringToOutput(node: Int, failure: Int): Unit = {
    val mergedOutput = output.getOrElse(node, 0) | output.getOrElse(failure, 0)
    if (mergedOutput > 0) {
      output(node) = mergedOutput
    }
  }

  private def findFailureState(state: Int, edgeIndex: Int): Int = {
    var failure = failureLink.getOrElse(state, -1)
    if (failure == -1) {
      return -1
    }

    // Find the deepest node labeled by proper suffix of String from root to current state.
    if (transitionTable(failure)(edgeIndex) == -1) {
      failure = failureLink.getOrElse(failure, -1)
      if (failure == -1) {
        return -1
      }
    }
    failure = transitionTable(failure)(edgeIndex)

    failure
  }

  /** Second step of Aho-Corasick algorithm: The algorithm starts at the input text’s beginning
    * and in the root state during searching for patterns. It processes the input string in a
    * single pass, and all occurrences of keywords are found, even if they overlap each other.
    * This implementation adds tokens parameter to match only exact tokens (not substrings)
    */
  def searchWords(sentence: Sentence, tokens: Map[Int, Annotation]): Seq[Annotation] = {
    var currentState = 0
    val longestMatchedTokens: ArrayBuffer[Annotation] = ArrayBuffer()

    sentence.content.zipWithIndex.foreach { case (char, textIndex) =>
      val currentChar = if (caseSensitive) char else char.toLower
      currentState = findNextState(currentState, currentChar)
      if (currentState > 0) {
        flattenEntityPatterns.zipWithIndex.foreach { case (flattenEntityPattern, keywordIndex) =>
          val bitwiseAndResult = output.getOrElse(currentState, 0) & 1 << keywordIndex
          if (bitwiseAndResult > 0) {
            val keyword = flattenEntityPattern.keyword
            val documentIndex = textIndex + sentence.start
            val begin = documentIndex - keyword.length + 1
            val candidateAnnotation = annotateTokenMatch(begin, sentence, flattenEntityPattern)

            if (foundToken(tokens, documentIndex, keyword)) {
              val overlappingTokens = longestMatchedTokens.filter(annotation =>
                annotation.begin >= begin & annotation.end <= documentIndex)

              overlappingTokens.foreach(overlappingToken =>
                longestMatchedTokens -= overlappingToken)

              if (!longestMatchedTokens.exists(annotation =>
                  annotation.end == candidateAnnotation.end)) {
                longestMatchedTokens.append(candidateAnnotation)
              }
            }
          }
        }
      }
    }

    longestMatchedTokens.toList
  }

  private def foundToken(tokens: Map[Int, Annotation], key: Int, keyword: String): Boolean = {
    val annotatedToken = tokens.get(key)
    val token =
      if (annotatedToken.isEmpty) ""
      else {
        if (caseSensitive) annotatedToken.get.result else annotatedToken.get.result.toLowerCase()
      }

    val currentKeyword = if (caseSensitive) keyword else keyword.toLowerCase()

    if (currentKeyword.split(" ").length == 1) {
      token == currentKeyword
    } else {
      currentKeyword.split(" ").contains(token)
    }
  }

  private def findNextState(state: Int, edge: Char): Int = {
    var currentState = state
    val currentEdge = if (caseSensitive) edge else edge.toLower

    val newLine = System.getProperty("line.separator")
    if (newLine == edge.toString) return 0

    val edgeIndex = edges.getOrElse(
      currentEdge,
      throw new UnsupportedOperationException(
        s"Char $currentEdge not found on alphabet. Please check alphabet"))

    // If goto is not defined, use failure function
    var nextState = transitionTable(currentState)(edgeIndex)
    if (nextState == -1) {
      currentState = failureLink.getOrElse(currentState, -1)
      if (currentState == -1) {
        return 0
      }
      nextState = transitionTable(currentState)(edgeIndex)
    }

    if (nextState == -1) 0 else nextState

  }

  private def annotateTokenMatch(
      start: Int,
      sentence: Sentence,
      flattenEntityPattern: FlattenEntityPattern): Annotation = {
    val keyword = flattenEntityPattern.keyword
    val matchedMetadata = if (flattenEntityPattern.id.isDefined) {
      Map("id" -> flattenEntityPattern.id.get, "entity" -> flattenEntityPattern.entity)
    } else {
      Map("entity" -> flattenEntityPattern.entity)
    }
    val end = start + keyword.length - 1

    Annotation(
      CHUNK,
      start,
      end,
      keyword,
      matchedMetadata ++ Map("sentence" -> sentence.index.toString))

  }

}
