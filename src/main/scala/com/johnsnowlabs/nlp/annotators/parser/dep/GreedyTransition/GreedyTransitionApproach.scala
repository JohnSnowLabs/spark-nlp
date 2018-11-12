package com.johnsnowlabs.nlp.annotators.parser.dep.GreedyTransition

import com.johnsnowlabs.nlp.annotators.common.{DependencyParsedSentence, WordWithDependency}
import com.johnsnowlabs.nlp.annotators.common.Annotated.PosTaggedSentence
import com.johnsnowlabs.nlp.annotators.parser.dep.{Perceptron, Tagger}
import com.sun.org.apache.bcel.internal.util.ClassVector

import scala.collection.mutable

/**
  * Parser based on the code of Matthew Honnibal and Martin Andrews
  */
class GreedyTransitionApproach {

  def parseInTraining(posTagged: PosTaggedSentence, trainedPerceptron: Array[String]): DependencyParsedSentence = {
    val dependencyMaker = loadPerceptronInTraining(trainedPerceptron)
    val sentence: Sentence = posTagged.indexedTaggedWords
      .map { item => WordData(item.word, item.tag) }.toList
    val dependencies = dependencyMaker.predictHeads(sentence)
    val words = posTagged.indexedTaggedWords
      .zip(dependencies)
      .map{
        case (word, dependency) =>
          WordWithDependency(word.word, word.begin, word.end, dependency)
      }

    DependencyParsedSentence(words)
  }

  def parseInPrediction(posTagged: PosTaggedSentence, trainedPerceptron: Array[String]): DependencyParsedSentence = {
    val dependencyMaker = loadPerceptronInPrediction(trainedPerceptron)
    val sentence: Sentence = posTagged.indexedTaggedWords
      .map { item => WordData(item.word, item.tag) }.toList
    val dependencies = dependencyMaker.predictHeads(sentence)
    val words = posTagged.indexedTaggedWords
      .zip(dependencies)
      .map{
        case (word, dependency) =>
          WordWithDependency(word.word, word.begin, word.end, dependency)
      }

    DependencyParsedSentence(words)
  }

  def loadPerceptronInTraining(trainedPerceptron: Array[String]): DependencyMaker = {
    val dependencyMaker = new DependencyMaker()
    dependencyMaker.perceptron.load(trainedPerceptron.toIterator)
    dependencyMaker.perceptron.resetLearning()
    dependencyMaker
  }

  def loadPerceptronInPrediction(trainedPerceptron: Array[String]): DependencyMaker = {
    val dependencyMaker = new DependencyMaker()
    dependencyMaker.perceptron.load(trainedPerceptron.toIterator)
    dependencyMaker
  }

  class DependencyMaker {
    val SHIFT: Move = 0
    val RIGHT: Move = 1
    val LEFT: Move = 2
    val INVALID: Move = -1

    val perceptron = new Perceptron(3)

    case class ParserState(n: Int, heads: Vector[Int], lefts: Vector[List[Int]], rights: Vector[List[Int]]) {

      def this(n: Int) = this(n, Vector.fill(n)(0: Int), Vector.fill(n + 1)(List[Int]()), Vector.fill(n + 1)(List[Int]()))

      def addArc(head: Int, child: Int): ParserState = {
        if (child < head) ParserState(n, heads.updated(child, head), lefts.updated(head, child :: lefts(head)), rights)
        else ParserState(n, heads.updated(child, head), lefts, rights.updated(head, child :: rights(head)))
      }
    }

    object ParserState {
      def apply(n: Int) = new ParserState(n)
    }

    def ParseStateInit(n:Int) = {
      // heads are the dependencies for each word in the sentence, except the last one (the ROOT)
      val heads = Vector.fill(n)(0:Int) // i.e. (0, .., n-1)

      // Each possible head (including ROOT) has a (lefts) and (rights) list, initially none
      // Entries (0, ..., n-1) are words, (n) is the 'ROOT'  ('to' is INCLUSIVE)
      val lefts  = (0 to n).map( i => List[Int]() ).toVector
      val rights = (0 to n).map( i => List[Int]() ).toVector
      ParserState(n, heads, lefts, rights)
    }

    case class CurrentState(i: Int, stack: List[Int], parse: ParserState) {
      def transition(move: Move): CurrentState = move match {
        case SHIFT => CurrentState(i+1, i :: stack, parse)
        case RIGHT => CurrentState(i, stack.tail, parse.addArc(stack.tail.head, stack.head))
        case LEFT  => CurrentState(i, stack.tail, parse.addArc(i, stack.head))
      }

      lazy val validMoves: Set[Move] = {
        ((if (i < parse.n) Some(SHIFT) else None) ::
        (if (stack.length >= 2) Some(RIGHT) else None) ::
        (if (stack.nonEmpty) Some(LEFT) else None) ::
        Nil).flatten.toSet
      }

      def getGoldMoves(goldHeads: Vector[Int]) = {
        def findDepsBetween(target: Int, others: List[Int]) = {
          others.exists( word => goldHeads(word)==target || goldHeads(target) == word)
        }

        if (stack.isEmpty || ( validMoves.contains(SHIFT) && goldHeads(i) == stack.head )) {
          Set(SHIFT)
        } else if ( goldHeads(stack.head) == i ) {
          Set(LEFT)
        } else {
          val leftIncorrect = stack.length >= 2 && goldHeads(stack.head) == stack.tail.head
          val dontPushI    = validMoves.contains(SHIFT) && findDepsBetween(i, stack)
          val dontPopStack = findDepsBetween(stack.head, ((i+1) until parse.n).toList)
          val nonGold = (
            (if (leftIncorrect) Some(LEFT) else None) ::
            (if (dontPushI) Some(SHIFT) else None) ::
            (if (dontPopStack)  Some(LEFT) else None) ::
            (if (dontPopStack)  Some(RIGHT) else None) ::
            Nil
          ).flatten.toSet

          validMoves -- nonGold
        }
      }

      def extractFeatures(words: Vector[Word], tags: Vector[ClassName]): Map[Feature, Score] = {
        def getStackContext[T <: String](data: Vector[T]): (T, T, T) = (
          if (stack.nonEmpty) data(stack.head) else "".asInstanceOf[T],
          if (stack.length > 1) data(stack(1)) else "".asInstanceOf[T],
          if (stack.length > 2) data(stack(2)) else "".asInstanceOf[T]
        )

        def getBufferContext[T <: String](data: Vector[T]): (T, T, T) = (
          if (i + 0 < parse.n) data(i + 0) else "".asInstanceOf[T],
          if (i + 1 < parse.n) data(i + 1) else "".asInstanceOf[T],
          if (i + 2 < parse.n) data(i + 2) else "".asInstanceOf[T]
        )

        def getParseContext[T <: String](idx: Int, deps: Vector[List[Int]], data: Vector[T]): (Int, T, T) = {
          if (idx < 0) {
            (0, "".asInstanceOf[T], "".asInstanceOf[T])
          } else {
            val dependencies = deps(idx)
            val valency = dependencies.length
            (
              valency,
              if (valency > 0) data(dependencies.head) else "".asInstanceOf[T],
              if (valency > 1) data(dependencies(1)) else "".asInstanceOf[T]
            )
          }
        }

        val n0 = i
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

    def train(sentences:List[Sentence], seed:Int, tagger: Tagger):Double = {
      val rand = new util.Random(seed)
      rand.shuffle(sentences).map( sentence =>trainOne(sentence, tagger) ).sum / sentences.length
    }

    def trainOne(sentence:Sentence, tagger: Tagger):Double = goodness(sentence, process(sentence, tagger))

    def goodness(sentence:Sentence, fit:List[Int]):Double = {
      val gold = sentence.map( _.dep ).toVector
      val correct = fit.zip( gold ).count( pair => pair._1 == pair._2)  / gold.length.toFloat
      //println(s"Dependency score : ${pct_fit_fmt_str(correct)}")
      correct
    }

    def process(sentence:Sentence, tagger: Tagger):List[Int] = {
      // NB: Our structure just has a 'pure' list of sentences.  The root will point to (n)
      // Previously it was assumed that the sentence has 1 entry pre-pended, and the stack starts at {1}

      // These should be Vectors, since we're going to be accessing them at random (not sequentially)
      val words      = sentence.map( _.norm ).toVector
      val tags       = tagger.tagSentence(sentence).toVector
      val gold_heads = sentence.map( _.dep ).toVector


      def moveThroughSentenceFrom(state: CurrentState): CurrentState = {
        val validMoves = state.validMoves
        if(validMoves.isEmpty) {
          state // This the answer!
        }
        else {
          val features = state.extractFeatures(words, tags)

          // This will produce scores for features that aren't valid too
          val score = perceptron.dotProductScore(features, perceptron.current)

          // Sort valid_moves scores into descending order, and pick the best move
          val guess = validMoves.map(m => (-score(m), m)).toList.minBy(_._1)._2

          val gold_moves = state.getGoldMoves(gold_heads)
          if(gold_moves.isEmpty) { throw new Exception(s"No Gold Moves at ${state.i}/${state.parse.n}!") }

          val best = gold_moves.map(m => (-score(m), m)).toList.minBy(_._1)._2
          perceptron.update(best, guess, features.keys)


          moveThroughSentenceFrom( state.transition(guess) )
        }
      }

      // This annotates the list of words so that parse.heads is its best guess when it finishes
      val finalState = moveThroughSentenceFrom( CurrentState(1, List(0), ParseStateInit(sentence.length)) )

      finalState.parse.heads.toList
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
          val guess = validMoves.map( m => (-score(m), m) ).toList.minBy { _._1 }._2
          moveThroughSentenceFrom( state.transition(guess) )
        }
      }

      val finalState = moveThroughSentenceFrom( CurrentState(1, List(0), ParserState(sentence.length)) )

      finalState.parse.heads.toList
    }

    def getPerceptronAsArray: Array[String] = {
      perceptron.toString().split("\\n")
    }

  }
}
