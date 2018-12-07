package com.johnsnowlabs.nlp.annotators.parser.dep.GreedyTransition

import com.johnsnowlabs.nlp.annotators.common.{DependencyParsedSentence, WordWithDependency}
import com.johnsnowlabs.nlp.annotators.common.Annotated.PosTaggedSentence
import com.johnsnowlabs.nlp.annotators.parser.dep.{Perceptron, Tagger}

/**
  * Parser based on the code of Matthew Honnibal and Martin Andrews
  */
class GreedyTransitionApproach {

  def parse(posTagged: PosTaggedSentence, trainedPerceptron: Array[String]): DependencyParsedSentence = {
    val dependencyMaker = loadPerceptronInPrediction(trainedPerceptron)
    val sentence: Sentence = posTagged.indexedTaggedWords
      .map { item => WordData(item.word, item.tag) }.toList
    getDependencies(sentence, dependencyMaker)
    val dependencies = dependencyMaker.predictHeads(sentence)
    val words = posTagged.indexedTaggedWords
      .zip(dependencies)
      .map{
        case (word, dependency) =>
          WordWithDependency(word.word, word.begin, word.end, dependency)
      }

    DependencyParsedSentence(words)
  }

  def loadPerceptronInPrediction(trainedPerceptron: Array[String]): DependencyMaker = {
    val dependencyMaker = new DependencyMaker()
    dependencyMaker.perceptron.load(trainedPerceptron.toIterator)
    dependencyMaker
  }

  def getDependencies(sentence: Sentence, dependencyMaker: DependencyMaker): List[Int] = {
    val dependencies = dependencyMaker.predictHeads(sentence)
    val headOccurrences = dependencies.groupBy(identity).mapValues(_.size)
    val numberOfPredictedRoots = headOccurrences.get(dependencies.size)

    if (numberOfPredictedRoots.get > 1) {
      println("numberOfPredictedRoots: " + numberOfPredictedRoots)
      //getDependencies(sentence, dependencyMaker)
    }
    dependencies
  }

  def loadPerceptronInTraining(trainedPerceptron: Array[String]): DependencyMaker = {
    val dependencyMaker = new DependencyMaker()
    dependencyMaker.perceptron.load(trainedPerceptron.toIterator)
    dependencyMaker.perceptron.cleanLearning()
    dependencyMaker
  }

  class DependencyMaker {
    val SHIFT: Move = 0
    val RIGHT: Move = 1
    val LEFT: Move = 2
    val INVALID: Move = -1

    val perceptron = new Perceptron(3)

    case class ParserState(sentenceLength: Int, heads: Vector[Int], lefts: Vector[List[Int]], rights: Vector[List[Int]]) {

      def this(sentenceLength: Int) = this(sentenceLength, Vector.fill(sentenceLength)(0: Int),
        Vector.fill(sentenceLength + 1)(List[Int]()),
        Vector.fill(sentenceLength + 1)(List[Int]()))

      def addArc(head: Int, child: Int, flag:String, move: Move): ParserState = {
        if (flag == "train"){
          if (child < head) {
            ParserState(sentenceLength, heads.updated(child, head), lefts.updated(head, child :: lefts(head)), rights)
          }
          else {
            ParserState(sentenceLength, heads.updated(child, head), lefts, rights.updated(head, child :: rights(head)))
          }
        } else {
          if (move == LEFT) {
            ParserState(sentenceLength, heads.updated(head-1, child-1), lefts.updated(head, child :: lefts(head)), rights)
          }
          else {
            ParserState(sentenceLength, heads.updated(child-1, head-1), lefts, rights.updated(head, child :: rights(head)))
          }
        }

      }
    }

    object ParserState {
      def apply(n: Int) = new ParserState(n)
    }

    def ParseStateInit(sentenceLength:Int) = {
      // heads are the dependencies for each word in the sentence, except the last one (the ROOT)
      val heads = Vector.fill(sentenceLength)(0:Int) // i.e. (0, .., n-1)

      // Each possible head (including ROOT) has a (lefts) and (rights) list, initially none
      // Entries (0, ..., n-1) are words, (n) is the 'ROOT'  ('to' is INCLUSIVE)
      val lefts  = (0 to sentenceLength).map(_ => List[Int]() ).toVector
      val rights = (0 to sentenceLength).map(_ => List[Int]() ).toVector
      ParserState(sentenceLength, heads, lefts, rights)
    }

    case class CurrentState(step: Int, stack: List[Int], parse: ParserState) {

      def transition(move: Move, flag: String): CurrentState = move match {
        //TODO: Verify correct values in stack after LFT or RIGHT
        case SHIFT => CurrentState(step+1, step :: stack, parse)
        case RIGHT => if (flag == "train") {
          CurrentState(step, stack.tail, parse.addArc(stack.tail.head, stack.head, flag, move))
        } else {
          val dependency = stack.take(2).head
          var newStack = stack.filter(_ != dependency)
          if (dependency == 0) {
            newStack = stack //Root is always on stack
          }
          CurrentState(step, newStack, parse.addArc(stack.tail.head, stack.head, flag, move))
        }
        case LEFT  => if (flag == "train") {
          CurrentState(step, stack.tail, parse.addArc(step, stack.head, flag, move))
        } else {
          val dependency = stack.take(2).last
          val newStack = stack.filter(_!=dependency)
          CurrentState(step, newStack, parse.addArc(stack.tail.head, stack.head, flag, move))
        }
      }

      lazy val validMovesOld: Set[Move] = {
        ((if (step < parse.sentenceLength) Some(SHIFT) else None) ::
          (if (stack.length >= 2) Some(RIGHT) else None) ::
          (if (stack.nonEmpty) Some(LEFT) else None) ::
          Nil).flatten.toSet
      }

      lazy val validMovesNew: Set[Move] = {
        (verifyShiftMove :: verifyRightMove :: verifyLeftMove :: Nil).flatten.toSet
      }

      def verifyShiftMove: Option[Move] = {
        var shiftMove: Option[Move] = Some(SHIFT)
        val processedWords = getNumberOfProcessedWords
        if (processedWords == parse.sentenceLength) {
          shiftMove = None
        }

        if (processedWords+stack.size-1 == parse.sentenceLength) {
          shiftMove = None //There are no more words to process
        }

        if (step > parse.sentenceLength) {
          shiftMove = None
        }

        shiftMove
      }

      def getNumberOfProcessedWords: Int = {
        val numberOfLeftWords = parse.lefts.flatten.size //parse.lefts.count(_.nonEmpty)
        val numberOfRightWords = parse.rights.flatten.size //parse.rights.count(_.nonEmpty)
        numberOfLeftWords+numberOfRightWords
      }

      def verifyRightMove: Option[Move] = {
        var rightMove: Option[Move] = Some(RIGHT)
        if (stack.length >= 2) {
          //          if (rightDependencyHasHead()) {
          //            rightMove = None //A dependency can have only one head
          //          } else {
          //            rightMove = Some(RIGHT)
          //          }
          rightMove = Some(RIGHT)
        } else{
          rightMove = None
        }

        if (rootHasDependency) {
          rightMove = None
        }

        if (stack.size == 2 && !isLastWord) { //Verifies if root can be assigned
          rightMove = None
        }

        rightMove
      }

      def rootHasDependency: Boolean = {
        parse.heads.contains(-1)
      }

      def isLastWord: Boolean = {
        val processedWords = getNumberOfProcessedWords
        processedWords == parse.sentenceLength-1
      }

      def verifyLeftMove: Option[Move] = {
        var leftMove: Option[Move] = Some(LEFT)

        if (stack.nonEmpty){
          if (stack.length == 1 && stack.head == 0) {
            leftMove = None
          }
          if (stack.length == 2 && stack.last == 0) {
            leftMove = None // Root cannot be a dependency
          }
          if (stack.length > 2){
            val leftDependency = stack.take(2).last
            if (leftDependency == 0) {
              leftMove = None //Root cannot have incoming arcs
            }

            //            if (leftDependencyHasHead()) {
            //              leftMove = None //A dependency can have only one head
            //            }

          }

        } else {
          leftMove = None
        }
        leftMove
      }

      //      def rightDependencyHasHead(): Boolean = {
      //        var hasHead = false
      //        val rightDependency = stack.take(2).head
      //        if (parse.lefts(rightDependency).nonEmpty || parse.rights(rightDependency).nonEmpty){
      //          hasHead = true
      //        }
      //        hasHead
      //      }
      //
      //      def leftDependencyHasHead(): Boolean = {
      //        var hasHead = false
      //        val leftDependency = stack.take(2).last
      //        if (parse.lefts(leftDependency).nonEmpty || parse.rights(leftDependency).nonEmpty){
      //          hasHead = true
      //        }
      //        hasHead
      //      }


      def getGoldMoves(goldHeads: Vector[Int]) = {
        def findDepsBetween(target: Int, others: List[Int]) = {
          others.exists( word => goldHeads(word)==target || goldHeads(target) == word)
        }

        if (stack.isEmpty || ( validMovesOld.contains(SHIFT) && goldHeads(step) == stack.head )) {
          Set(SHIFT)
        } else if ( goldHeads(stack.head) == step ) {
          Set(LEFT)
        } else {
          val leftIncorrect = stack.length >= 2 && goldHeads(stack.head) == stack.tail.head
          val dontPushI    = validMovesOld.contains(SHIFT) && findDepsBetween(step, stack)
          val dontPopStack = findDepsBetween(stack.head, ((step+1) until parse.sentenceLength).toList)
          val nonGold = (
            (if (leftIncorrect) Some(LEFT) else None) ::
              (if (dontPushI) Some(SHIFT) else None) ::
              (if (dontPopStack)  Some(LEFT) else None) ::
              (if (dontPopStack)  Some(RIGHT) else None) ::
              Nil
            ).flatten.toSet

          validMovesOld -- nonGold
        }
      }

      def extractFeatures(words: Vector[Word], tags: Vector[ClassName], flag: String): Map[Feature, Score] = {

        def getStackContext[T <: String](data: Vector[T]): (T, T, T) = {
          var firstElement = "".asInstanceOf[T]
          if (flag == "train"){
            firstElement = if (stack.nonEmpty) data(stack.head) else "".asInstanceOf[T]
          } else {
            if (stack.head == 0){
              firstElement = "".asInstanceOf[T]
            } else {
              firstElement = if (stack.nonEmpty) data(stack.head-1) else "".asInstanceOf[T]
            }
          }

          (firstElement,
            if (stack.length > 1) data(stack(1)) else "".asInstanceOf[T],
            if (stack.length > 2) data(stack(2)) else "".asInstanceOf[T]
          )
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
            var secondElement = "".asInstanceOf[T]
            if (flag == "train") {
              secondElement = if (valency > 0) data(dependencies.head) else "".asInstanceOf[T]
            } else {
              if (valency > 0 && dependencies.head == 0){
                secondElement = "".asInstanceOf[T]
              } else {
                secondElement = if (valency > 0) data(dependencies.head-1) else "".asInstanceOf[T]
              }
            }
            (
              valency,
              secondElement,
              if (valency > 1) data(dependencies(1)) else "".asInstanceOf[T]
            )
          }
        }

        var n0 = 0
        if (flag == "train"){
          n0 = step
        } else {
          n0 = stack.head
        }

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

    def trainOne(sentence:Sentence, tagger: Tagger):Double = goodness(sentence, processTrain(sentence, tagger))

    def goodness(sentence:Sentence, fit:List[Int]):Double = {
      val gold = sentence.map( _.dep ).toVector
      val correct = fit.zip( gold ).count( pair => pair._1 == pair._2)  / gold.length.toFloat
      //println(s"Dependency score : ${pct_fit_fmt_str(correct)}")
      correct
    }

    def processTrain(sentence:Sentence, tagger: Tagger):List[Int] = {
      // NB: Our structure just has a 'pure' list of sentences.  The root will point to (n)
      // Previously it was assumed that the sentence has 1 entry pre-pended, and the stack starts at {1}

      // These should be Vectors, since we're going to be accessing them at random (not sequentially)
      val words      = sentence.map( _.norm ).toVector
      val tags       = tagger.tagSentence(sentence).toVector
      val goldheads = sentence.map( _.dep ).toVector


      def moveThroughSentenceFrom(state: CurrentState): CurrentState = {
        val validMoves = state.validMovesOld
        if(validMoves.isEmpty) {
          state // This the answer!
        }
        else {
          val features = state.extractFeatures(words, tags, "train")

          // This will produce scores for features that aren't valid too
          val score = perceptron.dotProductScore(features, perceptron.current)

          // Sort valid_moves scores into descending order, and pick the best move
          val guess = validMoves.map(validMove => (-score(validMove), validMove)).toList.minBy(_._1)._2

          val goldMoves = state.getGoldMoves(goldheads)
          if(goldMoves.isEmpty) { throw new Exception(s"No Gold Moves at ${state.step}/${state.parse.sentenceLength}!") }

          val best = goldMoves.map(goldMove => (-score(goldMove), goldMove)).toList.minBy(_._1)._2
          perceptron.update(best, guess, features.keys)
          moveThroughSentenceFrom( state.transition(guess, "train") )
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
        val validMoves = state.validMovesNew
        if (validMoves.isEmpty) {
          state
        } else {
          //println("Heads 1: "+ state.parse.heads)
          //          if (state.step == 6){ //TODO: Remove
          //            println("Debug....")
          //          }
          val features = state.extractFeatures(words, tags, "prediction")
          //println("Heads 2: "+ state.parse.heads)
          val score = perceptron.score(features, perceptron.average)
          //val pruebas = validMoves.map( validMove => (-score(validMove), validMove) )
          //Choose the min score (_1) and gets the corresponding move (_2)
          val move = validMoves.map( validMove => (-score(validMove), validMove) ).toList.minBy { _._1 }._2
          println("Move: " + move)
          println("Heads Before transition: "+ state.parse.heads)
          println("Stack Before transition: "+ state.stack)
          //          println("Lefts Before transition:"+ state.parse.lefts)
          //          println("Rights Before transition:"+ state.parse.lefts)
          val nextState = state.transition(move, "prediction")
          println("Heads After transition: "+ nextState.parse.heads)
          println("Stack After transition: "+ nextState.stack)
          //          println("Lefts After transition:"+ nextState.parse.lefts)
          //          println("Rights After transition:"+ nextState.parse.lefts)
          moveThroughSentenceFrom(nextState)
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
