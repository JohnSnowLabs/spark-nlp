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

package com.johnsnowlabs.nlp.util

import org.scalatest.flatspec.AnyFlatSpec

class GraphBuilderTest extends AnyFlatSpec {

  "Graph Builder" should "initialize a graph with 5 vertices" in {
    val numberOfVertices = 5

    val graph = new GraphBuilder(numberOfVertices)

    assert(graph.getNumberOfVertices == numberOfVertices)
  }

  it should "raise an error when setting wrong number of vertices" in {
    val numberOfVertices = -1

    assertThrows[IllegalArgumentException] {
      new GraphBuilder(numberOfVertices)
    }
  }

  it should "add an edge to the graph" in {
    val numberOfVertices = 5
    val graph = new GraphBuilder(numberOfVertices)
    val expectedAdjacentVertices = Set(1, 4)

    graph.addEdge(0, 1)
    graph.addEdge(0, 4)
    val actualAdjacentVertices = graph.getAdjacentVertices(0)

    assert(actualAdjacentVertices == expectedAdjacentVertices)
  }

  it should "raise an error when adding an edge for a source that does not exist" in {
    val numberOfVertices = 3
    val graph = new GraphBuilder(numberOfVertices)

    assertThrows[NoSuchElementException] {
      graph.addEdge(4, 1)
    }

  }

  it should "raise an error when adding an edge for a destination that does not exist" in {
    val numberOfVertices = 3
    val graph = new GraphBuilder(numberOfVertices)

    assertThrows[NoSuchElementException] {
      graph.addEdge(1, 4)
    }

  }

  it should "return true if an edge exists" in {
    val numberOfVertices = 5
    val graph = new GraphBuilder(numberOfVertices)
    graph.addEdge(1, 4)
    graph.addEdge(3, 2)

    assert(graph.edgeExists(1, 4))
    assert(graph.edgeExists(3, 2))
  }

  it should "return false if an edge does not exist" in {
    val numberOfVertices = 3
    val graph = new GraphBuilder(numberOfVertices)
    graph.addEdge(0, 1)

    assert(!graph.edgeExists(1, 1))
  }

  it should "return paths from root vertex of a graph" in {
    val graph = getTestGraph
    val expectedPaths = List(List(0, 1, 3), List(0, 2, 5), List(0, 1, 4))

    val actualPaths = List(3, 5, 4).map(i => graph.findPath(0, i))

    assert(expectedPaths == actualPaths)
  }

  it should "return paths from any vertex of a graph" in {
    val graph = getTestGraph
    val expectedPaths = List(List(2, 5), List(1, 4, 6))

    val actualPaths = List((2, 5), (1, 6)).map(i => graph.findPath(i._1, i._2))

    assert(expectedPaths == actualPaths)
  }

  it should "return empty path when there is no path between two vertices" in {
    val graph = getTestGraph

    val actualPath = graph.findPath(3, 5)

    assert(actualPath.isEmpty)

  }

  it should "raise an error when trying to find paths on graph without edges" in {
    val graph = new GraphBuilder(1)

    assertThrows[IllegalArgumentException] {
      graph.findPath(4, 7)
    }
  }

  private def getTestGraph: GraphBuilder = {
    val numberOfVertices = 7
    val graph = new GraphBuilder(numberOfVertices)
    graph.addEdge(0, 1)
    graph.addEdge(0, 2)
    graph.addEdge(1, 3)
    graph.addEdge(1, 4)
    graph.addEdge(2, 5)
    graph.addEdge(4, 6)

    graph
  }

}
