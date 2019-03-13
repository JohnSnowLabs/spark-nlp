package com.johnsnowlabs.nlp.annotators.parser.dep.GreedyTransition

import com.johnsnowlabs.nlp.annotators.parser.dep.Perceptron

class DependencyMakerPrediction {

  val SHIFT: Move = 0
  val RIGHT: Move = 1
  val LEFT: Move = 2

  val perceptron = new Perceptron(3)

  case class ParserState(sentenceLength: Int, heads: Vector[Int], lefts: Vector[List[Int]], rights: Vector[List[Int]]) {

    def this(sentenceLength: Int) = this(sentenceLength, Vector.fill(sentenceLength)(0: Int),
      Vector.fill(sentenceLength + 1)(List[Int]()),
      Vector.fill(sentenceLength + 1)(List[Int]()))

    def addArc(head: Int, child: Int, move: Move): ParserState = {
      if (move == LEFT) {
        ParserState(sentenceLength, heads.updated(head-1, child-1), lefts.updated(head, child :: lefts(head)), rights)
      }
      else {
        ParserState(sentenceLength, heads.updated(child-1, head-1), lefts, rights.updated(head, child :: rights(head)))
      }
    }
  }

  object ParserState {
    def apply(n: Int) = new ParserState(n)
  }

  case class CurrentState(step: Int, stack: List[Int], parse: ParserState) {

    def transition(move: Move): CurrentState = move match {

      case SHIFT => CurrentState(step+1, step :: stack, parse)
      case RIGHT =>
        val dependency = stack.take(2).head
        var newStack = stack.filter(_ != dependency)
        if (dependency == 0) {
          newStack = stack //Root is always on stack
        }
        CurrentState(step, newStack, parse.addArc(stack.tail.head, stack.head, move))
      case LEFT  =>
        val dependency = stack.take(2).last
        val newStack = stack.filter(_!=dependency)
        CurrentState(step, newStack, parse.addArc(stack.tail.head, stack.head, move))
    }

    lazy val validMoves: Set[Move] = {
      (verifyShiftMove :: verifyRightMove :: verifyLeftMove :: Nil).flatten.toSet
    }

    def verifyShiftMove: Option[Move] = {
      var shiftMove: Option[Move] = Some(SHIFT)
      val processedWords = getNumberOfProcessedWords

      if (processedWords == parse.sentenceLength) {
        shiftMove = None
      }

      if (processedWords+stack.size-1 == parse.sentenceLength) {
        shiftMove = None //There are no more words to take
      }

      shiftMove
    }

    def getNumberOfProcessedWords: Int = {
      val numberOfLeftWords = parse.lefts.flatten.size
      val numberOfRightWords = parse.rights.flatten.size
      numberOfLeftWords+numberOfRightWords
    }

    def verifyRightMove: Option[Move] = {
      var rightMove: Option[Move] = Some(RIGHT)

      if (stack.length >= 2) {
        rightMove = Some(RIGHT)
      } else{
        rightMove = None
      }

      if (stack.size == 2 && !isLastWord) { //Verifies if root can be assigned
        rightMove = None
      }

      rightMove
    }

    def isLastWord: Boolean = {
      val processedWords = getNumberOfProcessedWords
      processedWords == parse.sentenceLength-1
    }

    def verifyLeftMove: Option[Move] = {
      var leftMove: Option[Move] = Some(LEFT)

      if (stack.nonEmpty && !isValidRootMove){
        leftMove = None
      } else {
        leftMove = None
      }
      leftMove
    }

    def isValidRootMove: Boolean = {
      var valid = true

      if (stack.length == 1 && stack.head == 0) {
        valid = false
      }

      if (stack.length == 2 && stack.last == 0) {
        valid = false // Root cannot be a dependency
      }

      if (stack.length > 2){
        val leftDependency = stack.take(2).last
        if (leftDependency == 0) {
          valid = false //Root cannot have incoming arcs
        }
      }
      valid
    }


    def extractFeatures(words: Vector[Word], tags: Vector[ClassName]): Map[Feature, Score] = {

      def getStackContext[T <: String](data: Vector[T]): (T, T, T) = {
        val firstElement = getFirstElement(data)
        (firstElement,
          if (stack.length > 1) data(stack(1)) else "".asInstanceOf[T],
          if (stack.length > 2) data(stack(2)) else "".asInstanceOf[T]
        )
      }

      def getFirstElement[T <: String](data: Vector[T]): T = {
        var firstElement = "".asInstanceOf[T]
        if (stack.head != 0){
          firstElement = if (stack.nonEmpty) data(stack.head-1) else "".asInstanceOf[T]
        }
        firstElement
      }

      def getBufferContext[T <: String](data: Vector[T]): (T, T, T) = (
        if (step + 0 < parse.sentenceLength) data(step + 0) else "".asInstanceOf[T],
        if (step + 1 < parse.sentenceLength) data(step + 1) else "".asInstanceOf[T],
        if (step + 2 < parse.sentenceLength) data(step + 2) else "".asInstanceOf[T]
      )

      def getParseContext[T <: String](idx: Int, deps: Vector[List[Int]], data: Vector[T]): (Int, T, T) = {
        if (idx < 0) {
          (0, "".asInstanceOf[T], "".asInstanceOf[T])
        } else {
          val dependencies = deps(idx)
          val valency = dependencies.length
          val secondElement = getSecondElement(dependencies, data, valency)
          (
            valency,
            secondElement,
            if (valency > 1) data(dependencies(1)) else "".asInstanceOf[T]
          )
        }
      }

      def getSecondElement[T <: String](dependencies: List[Int], data: Vector[T], valency: Int): T = {
        var secondElement = "".asInstanceOf[T]

        if (valency > 0 && dependencies.head == 0){
          secondElement = "".asInstanceOf[T]
        } else {
          secondElement = if (valency > 0) data(dependencies.head-1) else "".asInstanceOf[T]
        }

        secondElement
      }

      val n0 = stack.head

      val s0 = if (stack.isEmpty) -1 else stack.head

      val (ws0, ws1, ws2) = getStackContext(words)
      val (ts0, ts1, ts2) = getStackContext(tags)

      val (wn0, wn1, wn2) = getBufferContext(words)
      val (tn0, tn1, tn2) = getBufferContext(tags)

      val (vn0b, wn0b1, wn0b2) = getParseContext(n0, parse.lefts,  words)
      val (_   , tn0b1, tn0b2) = getParseContext(n0, parse.lefts,  tags)

      val (vn0f, wn0f1, wn0f2) = getParseContext(n0, parse.rights, words)
      val (_,    tn0f1, tn0f2) = getParseContext(n0, parse.rights, tags)

      val (vs0b, ws0b1, ws0b2) = getParseContext(s0, parse.lefts,  words)
      val (_,    ts0b1, ts0b2) = getParseContext(s0, parse.lefts,  tags)

      val (vs0f, ws0f1, ws0f2) = getParseContext(s0, parse.rights, words)
      val (_,    ts0f1, ts0f2) = getParseContext(s0, parse.rights, tags)

      val dist = if (s0 >= 0) math.min(n0 - s0, 5) else 0

      val bias = Feature("bias", "")

      val wordUnigrams = for(
        word <- List(wn0, wn1, wn2, ws0, ws1, ws2, wn0b1, wn0b2, ws0b1, ws0b2, ws0f1, ws0f2)
        if word.nonEmpty
      ) yield Feature("w", word)

      val tagUnigrams = for(
        tag  <- List(tn0, tn1, tn2, ts0, ts1, ts2, tn0b1, tn0b2, ts0b1, ts0b2, ts0f1, ts0f2)
        if tag.nonEmpty
      ) yield Feature("t", tag)

      val wordTagPairs = for(
        ((word, tag), idx) <- List((wn0, tn0), (wn1, tn1), (wn2, tn2), (ws0, ts0)).zipWithIndex
        if word.nonEmpty || tag.nonEmpty
      ) yield Feature(s"wt$idx", s"w=$word t=$tag")

      val bigrams = Set(
        Feature("w s0n0", s"$ws0 $wn0"),
        Feature("t s0n0", s"$ts0 $tn0"),
        Feature("t n0n1", s"$tn0 $tn1")
      )

      val trigrams = Set(
        Feature("wtw nns", s"$wn0/$tn0 $ws0"),
        Feature("wtt nns", s"$wn0/$tn0 $ts0"),
        Feature("wtw ssn", s"$ws0/$ts0 $wn0"),
        Feature("wtt ssn", s"$ws0/$ts0 $tn0")
      )

      val quadgrams = Set(
        Feature("wtwt", s"$ws0/$ts0 $wn0/$tn0")
      )

      val tagTrigrams = for(
        ((t0,t1,t2), idx) <- List( (tn0, tn1, tn2),     (ts0, tn0, tn1),     (ts0, ts1, tn0),    (ts0, ts1, ts1),
          (ts0, ts0f1, tn0),   (ts0, ts0f1, tn0),   (ts0, tn0, tn0b1),
          (ts0, ts0b1, ts0b2), (ts0, ts0f1, ts0f2),
          (tn0, tn0b1, tn0b2)
        ).zipWithIndex
        if t0.nonEmpty || t1.nonEmpty || t2.nonEmpty
      ) yield Feature(s"ttt-$idx", s"$t0 $t1 $t2")

      val valencyAndDistance = for(
        ((str, v), idx) <- List( (ws0, vs0f), (ws0, vs0b), (wn0, vn0b),
          (ts0, vs0f), (ts0, vs0b), (tn0, vn0b),
          (ws0, dist), (wn0, dist), (ts0, dist), (tn0, dist),
          ("t"+tn0+ts0, dist), ("w"+wn0+ws0, dist)
        ).zipWithIndex
        if str.nonEmpty || v != 0
      ) yield Feature(s"val$idx", s"$str $v")

      val featureSetCombined = Set(bias) ++ bigrams ++ trigrams ++ quadgrams ++
        wordUnigrams.toSet ++ tagUnigrams.toSet ++ wordTagPairs.toSet ++
        tagTrigrams.toSet ++ valencyAndDistance.toSet

      featureSetCombined.map( f => (f, 1: Score) ).toMap
    }

  }

  def predictHeads(sentence: Sentence): List[Int] = {
    val words = sentence.map( _.norm ).toVector
    val tags = sentence.map(s => s.pos).toVector

    def moveThroughSentenceFrom(state: CurrentState): CurrentState = {
      val validMoves = state.validMoves
      if (validMoves.isEmpty) {
        state
      } else {

        val features = state.extractFeatures(words, tags)
        val score = perceptron.score(features, perceptron.average)
        //Choose the min score (_1) and gets the corresponding move (_2)
        val move = validMoves.map( validMove => (-score(validMove), validMove) ).toList.minBy { _._1 }._2
        val nextState = state.transition(move)
        moveThroughSentenceFrom(nextState)
      }
    }

    val finalState = moveThroughSentenceFrom( CurrentState(1, List(0), ParserState(sentence.length)) )

    finalState.parse.heads.toList
  }

}
