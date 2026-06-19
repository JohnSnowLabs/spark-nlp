#  Copyright 2017-2024 John Snow Labs
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
import os
import unittest

import pytest

from sparknlp.annotator import *
from sparknlp.base import *
from test.util import SparkContextForTest

# Local ONNX export of a SaT model (model.onnx + assets/sentencepiece.bpe.model). Mirrors the Scala
# spec which loads from the folder "1" at the repository root; override with SAT_MODEL_PATH.
SAT_MODEL_PATH = os.environ.get("SAT_MODEL_PATH", "../1")

# Long English / German passages (>500 words each), transcribed verbatim from the Scala spec
# (SentenceDetectorSaTSpec) so both test suites exercise identical multi-window stitching.
LONG_ENGLISH = """The landscape of modern software engineering is undergoing a profound paradigm shift. For
decades, imperative and object-oriented programming methodologies held an undisputed monopoly
over the industry. Developers conceptualized software as a series of sequential instructions that
manipulated mutable state, a mental model that closely mirrored the underlying von Neumann
architecture of physical hardware. However, as the digital age advanced into the third decade
of the twenty-first century, the nature of computing hardware and the scale of software systems
changed dramatically. The relentless march of Moore's Law, which historically guaranteed steady
increases in single-core processor speeds, effectively reached its physical limitations. In response,
hardware manufacturers pivoted toward multi-core architectures, distributed systems, and massive
cloud computing environments. This fundamental shift exposed severe vulnerabilities in traditional
imperative paradigms, particularly regarding concurrency, parallelism, and state management.
Consequently, functional programming—a paradigm rooted in mathematical logic and developed
decades ago in academic isolation—emerged from the fringes to become a dominant force in
contemporary software architecture.

To understand the sudden ascendancy of functional programming, one must first dissect its core
philosophical tenets and contrast them with the traditional imperative model. At its heart,
imperative programming is preoccupied with the "how" of computation. It relies heavily on
statements that alter a program’s state, utilizing loops, conditional branches, and mutable
variables to achieve a desired outcome. While intuitive for simple tasks, this approach becomes
exponentially complex as systems scale. When multiple threads of execution attempt to read and
write to the same memory location simultaneously, they introduce a catastrophic category of bugs
known as race conditions. Debugging these transient, non-deterministic errors in a massive
object-oriented codebase can consume countless engineering hours and compromise system reliability.
Functional programming, conversely, is preoccupied with the "what" of computation. It treats
software execution as the evaluation of mathematical functions and explicitly avoids changing state
and mutable data. By adopting a declarative posture, functional programming shifts the focus
from step-by-step instruction execution to the transformation of data flowing through pure pipelines.

The bedrock upon which functional programming stands is the concept of immutability. In a purely
functional environment, once a variable, object, or data structure is created, it cannot be modified.
If a change is required, a new data structure is generated reflecting the updated state, leaving
the original completely intact. To developers deeply entrenched in object-oriented habits, this
initially appears wildly inefficient. The instinctive reaction is to worry about memory overhead,
garbage collection pressure, and performance degradation caused by constantly copying data.
However, modern language runtimes and advanced compiler optimizations have largely mitigated these
concerns through persistent data structures and structural sharing. When a new version of a tree
or map is created, it shares the vast majority of its internal nodes with the original instance,
making the operation remarkably lightweight. The architectural benefits of immutability far
outweigh its minor overhead. Immutability completely eliminates the threat of data races; if data
cannot change, it can be safely shared among an infinite number of concurrent threads without the
need for complex, performance-choking locking mechanisms like mutexes or semaphores.

Closely tied to immutability is the principle of purity, specifically the deployment of pure
functions. A function is deemed pure if it satisfies two rigorous mathematical criteria: it must Always
return the exact same output when presented with the same input, and it must possess absolutely no
side effects. A side effect is defined as any modification of state outside the local environment of the
function itself. This includes obvious actions like modifying a global variable, writing to a database,
or altering an input argument, as well as more subtle operations like logging to a console, throwing an
exception, or reading the current system time. In an imperative codebase, functions frequently hide
their true dependencies and behaviors, acting as black boxes that interact unpredictably with global
application state. Pure functions, by contrast, are completely transparent. This quality, known as
referential transparency, means that a function call can be safely replaced with its resulting value
without altering the program's behavior. This predictability vastly simplifies unit testing, eliminates
entire classes of runtime bugs, and allows compilers to aggressively optimize code via techniques
like memoization and lazy evaluation.

Furthermore, functional programming elevates functions to the status of first-class citizens. This
means that functions are treated no differently than primitive data types like integers or strings.
They can be assigned to variables, passed as arguments to other functions, and returned as values from
computations. This capability gives rise to higher-order functions, which serve as the primary engines
of code reuse and abstraction in functional paradigms. Classic examples such as map, filter, and reduce
allow developers to manipulate collections of data with unprecedented conciseness and clarity. Instead
of writing convoluted nested loops with manual indexing and temporary accumulator variables, a developer
can express complex data transformations in a single, readable chain of operations. This declarative
style filters out the boilerplate, allowing the underlying business logic to shine through. It also
abstracts away the mechanics of iteration, enabling the underlying framework to automatically split
the data across multiple processor cores and execute the transformation in parallel without any
modifications to the user's code.

As software systems have grown to global proportions, handling failures and asynchronous operations
gracefully has become a paramount concern. Traditional programming paradigms rely heavily on exceptions
for error handling, a practice that disrupts the normal control flow of an application and introduces
hidden exit points that are difficult to track. Functional programming introduces a elegant alternative
by encapsulating errors and optionality directly into the type system using algebraic data types
such as Option, Either, and Try. Instead of returning a null pointer or throwing a runtime exception
when a user is not found in a database, a functional system returns an Option type that explicitly
forces the developer to handle both the presence and absence of data at compile time. Similarly, asynchronous
computations and future events are wrapped in monadic containers like Futures or IO monads. These structures
allow developers to compose complex networks of asynchronous tasks using the exact same functional semantics
they use for standard data collections, creating a unified, robust framework for building resilient,
fault-tolerant, and highly responsive cloud-native applications.

The practical implications of these functional concepts are vividly illustrated by the evolution of
popular mainstream programming languages. For many years, a strict ideological divide existed between
purely functional languages like Haskell and object-oriented behemoths like Java and C++. However, the
pragmatic demands of modern software development have forced a convergence. Today, almost every major
imperative language has eagerly adopted functional features. Java introduced lambdas and streams;
JavaScript popularized first-class functions and immutable array patterns; and C++ integrated lambdas
and ranges. Perhaps the most compelling manifestation of this synthesis is found in hybrid languages
like Scala. Scala was intentionally designed to bridge the gap, seamlessly fusing object-oriented
programming with sophisticated, cutting-edge functional paradigms. This hybrid approach allows engineering
teams to leverage the familiar organizational patterns of classes and packages while fully exploiting
the mathematical safety and concurrent power of pure functional logic, proving that the paradigms are
not mutually exclusive but deeply complementary.

Ultimately, the widespread adoption of functional programming represents a maturation of the software
engineering discipline. As our reliance on distributed cloud environments, real-time data streaming, and
massively parallel processing continues to expand, the historical imperative approach of micro-managing CPU
registers and mutable memory locations becomes increasingly untenable. Functional programming offers a
higher level of abstraction that aligns perfectly with the architectural demands of the modern era. By
championing immutability, pure functions, referential transparency, and strong type systems, it provides
developers with the conceptual tools necessary to reason about complex systems deterministically. While the
initial learning curve can be steep for those habituated to imperative models, the long-term rewards are
undeniable: cleaner codebases, vastly reduced debugging cycles, seamless concurrency, and intrinsically
more reliable software. Functional programming has successfully transitioned from an esoteric academic
pursuit into a fundamental pillar of modern technology, permanently reshaping the way we conceptualize,
design, and execute software across the globe."""

LONG_GERMAN = """Die Landschaft der modernen Softwareentwicklung befindet sich in einem tiefgreifenden Paradigmenwechsel.
Über Jahrzehnte hinweg hielten imperative und objektorientierte Programmiermethoden ein unbestrittenes Monopol
in der Industrie. Entwickler konzipierten Software als eine Abfolge sequentieller Anweisungen, die einen
veränderlichen Zustand manipulierten – ein mentales Modell, das die zugrunde liegende Von-Neumann-Architektur
der physischen Hardware exakt widerspiegelte. Als das digitale Zeitalter jedoch in das dritte Jahrzehnt des
einundzwanzigsten Jahrhunderts vorschritt, änderten sich die Natur der Computerhardware und die Skalierung von
Softwaresystemen dramatisch. Der unerbittliche Marsch des Mooreschen Gesetzes, der historisch gesehen stetige
Steigerungen der Single-Core-Prozessorgeschwindigkeiten garantierte, stieß endgültig an seine physischen Grenzen.
Als Reaktion darauf schwenkten die Hardwarehersteller auf Multi-Core-Architekturen, verteilte Systeme und massive
Cloud-Computing-Umgebungen um. Dieser fundamentale Wandel legte gravierende Schwachstellen in traditionellen
imperativen Paradigmen offen, insbesondere im Hinblick auf Nebenläufigkeit, Parallelität und Zustandsverwaltung.
Infolgedessen trat die funktionale Programmierung – ein Paradigma, das in der mathematischen Logik verwurzelt ist
und vor Jahrzehnten in akademischer Isolation entwickelt wurde – aus dem Abseits hervor, um zu einer dominierenden
Kraft in der zeitgenössischen Softwarearchitektur zu werden.

Um den plötzlichen Aufstieg der funktionalen Programmierung zu verstehen, muss man zunächst ihre philosophischen
Kernpunkte sezieren und sie dem traditionellen imperativen Modell gegenüberstellen. Im Kern befasst sich die
imperative Programmierung mit dem „Wie“ der Berechnung. Sie stützt sich stark auf Anweisungen, die den Zustand
eines Programms verändern, und nutzt Schleifen, bedingte Verzweigungen und veränderliche Variablen, um ein
gewünschtes Ergebnis zu erzielen. Was für einfache Aufgaben intuitiv ist, wird exponentiell komplexer, sobald
Systeme skalieren. Wenn mehrere Ausführungsthreads gleichzeitig versuchen, denselben Speicherort zu lesen und zu
beschreiben, führen sie eine katastrophale Kategorie von Fehlern ein, die als Race Conditions (Wettlaufeffekte)
bekannt sind. Das Debuggen dieser flüchtigen, nicht-deterministischen Fehler in einer massiven objektorientierten
Codebasis kann unzählige Entwicklungsstunden verschlingen und die Systemzuverlässigkeit gefährden. Die funktionale
Programmierung hingegen befasst sich mit dem „Was“ der Berechnung. Sie behandelt die Programmausführung als die
Auswertung mathematischer Funktionen und vermeidet explizit die Änderung von Zuständen und veränderlichen Daten.
Durch diese deklarative Haltung verlagert die funktionale Programmierung den Fokus von der schrittweisen Ausführung
von Befehlen auf die Transformation von Daten, die durch reine Pipelines fließen.

Das Fundament, auf dem die funktionale Programmierung steht, ist das Konzept der Unveränderlichkeit (Immutability).
In einer rein funktionalen Umgebung kann eine Variable, ein Objekt oder eine Datenstruktur nach ihrer Erstellung nicht
mehr modifiziert werden. Wenn eine Änderung erforderlich ist, wird eine neue Datenstruktur erzeugt, die den aktualisierten
Zustand widerspiegelt, während das Original vollständig intakt bleibt. Auf Entwickler, die tief in objektorientierten
Gewohnheiten verwurzelt sind, wirkt dies anfangs oft absurd ineffizient. Die instinktive Reaktion ist die Sorge vor
Speicher-Overhead, Druck auf den Garbage Collector und Performance-Einbußen, die durch das ständige Kopieren von Daten
entstehen. Moderne Laufzeitumgebungen und fortschrittliche Compiler-Optimierungen haben diese Bedenken jedoch weitgehend
durch persistente Datenstrukturen und strukturelles Teilen (Structural Sharing) entkräftet. Wenn eine neue Version eines
Baums oder einer Map erstellt wird, teilt sie den weitaus größten Teil ihrer internen Knoten mit der ursprünglichen Instanz,
was die Operation bemerkenswert leichtgewichtig macht. Die architektonischen Vorteile der Unveränderlichkeit überwiegen den
minimalen Overhead bei Weitem. Unveränderlichkeit eliminiert die Bedrohung durch Daten-Wettläufe vollständig; wenn Daten
sich nicht ändern können, können sie sicher von einer unendlichen Anzahl von Threads gleichzeitig genutzt werden, ohne dass
komplexe, performance-mindernde Sperrmechanismen wie Mutexe oder Semaphore erforderlich sind.

Eng mit der Unveränderlichkeit verbunden ist das Prinzip der Reinheit, insbesondere der Einsatz reiner Funktionen. Eine
Funktion gilt als rein, wenn sie zwei strenge mathematische Kriterien erfüllt: Sie muss bei gleichen Eingaben immer exakt
dieselben Ausgaben zurückgeben, und sie darf absolut keine Nebeneffekte (Side Effects) besitzen. Ein Nebeneffekt ist definiert
als jede Modifikation des Zustands außerhalb der lokalen Umgebung der Funktion selbst. Dazu gehören offensichtliche Aktionen
wie das Ändern einer globalen Variable, das Schreiben in eine Datenbank oder das Verändern eines Eingabearguments, aber auch
subtilere Operationen wie das Protokollieren in einer Konsole, das Werfen einer Exception oder das Auslesen der aktuellen
Systemzeit. In einer imperativen Codebasis verbergen Funktionen häufig ihre wahren Abhängigkeiten und Verhaltensweisen und
agieren als Black Boxes, die unvorhersehbar mit dem globalen Anwendungszustand interagieren. Reine Funktionen hingegen sind
völlig transparent. Diese Qualität, bekannt als referenzielle Transparenz, bedeutet, dass ein Funktionsaufruf sicher durch
seinen resultierenden Wert ersetzt werden kann, ohne das Verhalten des Programms zu verändern. Diese Vorhersagbarkeit
vereinfacht Unit-Tests drastisch, eliminiert ganze Klassen von Laufzeitfehlern und ermöglicht es Compilern, Code durch Techniken
wie Memoisation und träge Auswertung (Lazy Evaluation) aggressiv zu optimieren.

Darüber hinaus erhebt die funktionale Programmierung Funktionen in den Status von Bürgern erster Klasse (First-Class Citizens).
Das bedeutet, dass Funktionen nicht anders behandelt werden als primitive Datentypen wie Integer oder Strings. Sie können
Variablen zugewiesen, als Argumente an andere Funktionen übergeben und als Ergebnisse aus Berechnungen zurückgegeben werden.
Diese Fähigkeit führt zu Funktionen höherer Ordnung (Higher-Order Functions), die als primäre Werkzeuge für Code-Wiederverwendung
und Abstraktion in funktionalen Paradigmen dienen. Klassische Beispiele wie map, filter und reduce ermöglichen es Entwicklern,
Datensammlungen mit beispielloser Prägnanz und Klarheit zu manipulieren. Anstatt verschachtelte Schleifen mit manueller Indizierung
und temporären Akkumulatorvariablen zu schreiben, kann ein Entwickler komplexe Datentransformationen in einer einzigen, lesbaren
Kette von Operationen ausdrücken. Dieser deklarative Stil filtert den Boilerplate-Code heraus, sodass die zugrunde liegende
Geschäftslogik klar hervortritt. Zudem abstrahiert er die Mechanik der Iteration, was es dem zugrunde liegenden Framework erlaubt,
die Daten automatisch auf mehrere Prozessorkerne aufzuteilen und die Transformation parallel auszuführen, ohne dass der Code des
Entwicklers dafür modifiziert werden muss.

Da Softwaresysteme mittlerweile globale Ausmaße angenommen haben, ist der elegante Umgang mit Fehlern und asynchronen Operationen
zu einem zentralen Anliegen geworden. Traditionelle Programmierparadigmen verlassen sich beim Error-Handling stark auf Exceptions,
eine Praxis, die den normalen Kontrollfluss einer Anwendung unterbricht und versteckte Ausstiegspunkte einführt, die schwer zu
verfolgen sind. Die funktionale Programmierung führt eine elegante Alternative ein, indem sie Fehler und Optionalität mithilfe
algebraischer Datentypen wie Option, Either und Try direkt in das Typsystem kapselt. Anstatt einen Null-Pointer zurückzugeben oder
eine Laufzeit-Exception zu werfen, wenn ein Benutzer in einer Datenbank nicht gefunden wird, gibt ein funktionales System einen
Option-Typ zurück. Dieser zwingt den Entwickler explizit dazu, sowohl das Vorhandensein als auch das Fehlen von Daten bereits zur
Kompilierzeit zu behandeln. Ähnlich werden asynchrone Berechnungen und zukünftige Ereignisse in monadische Container wie Futures oder
IO-Monaden verpackt. Diese Strukturen erlauben es Entwicklern, komplexe Netzwerke asynchroner Aufgaben mit exakt derselben funktionalen
Semantik zu komponieren, die sie für Standard-Datensammlungen verwenden. Dadurch entsteht ein einheitliches, robustes Framework für
den Bau resilienter, fehlertoleranter und hochgradig reaktiver cloud-nativer Anwendungen.

Die praktischen Auswirkungen dieser funktionalen Konzepte werden durch die Evolution moderner Mainstream-Programmiersprachen lebhaft
illustriert. Viele Jahre lang existierte eine strikte ideologische Kluft zwischen rein funktionalen Sprachen wie Haskell und
objektorientierten Giganten wie Java und C++. Die pragmatischen Anforderungen der modernen Softwareentwicklung haben jedoch eine
Konvergenz erzwungen. Heute hat fast jede größere imperative Sprache eifrig funktionale Features adoptiert. Java führte Lambdas und
Streams ein; JavaScript popularisierte First-Class-Functions und unveränderliche Array-Muster; und C++ integrierte Lambdas und Ranges.
Die wohl überzeugendste Manifestation dieser Synthese findet sich in Hybridsprachen wie Scala. Scala wurde bewusst entwickelt, um diese
Lücke zu schließen, indem es objektorientierte Programmierung nahtlos mit hochentwickelten, modernen funktionalen Paradigmen verschmilzt.
Dieser hybride Ansatz ermöglicht es Entwicklungsteams, die vertrauten Organisationsmuster von Klassen und Paketen zu nutzen, während sie
gleichzeitig die mathematische Sicherheit und die parallele Leistungsfähigkeit rein funktionaler Logik voll ausschöpfen. Dies beweist,
dass die Paradigmen sich keineswegs gegenseitig ausschließen, sondern tiefgreifend ergänzen.

Letztendlich repräsentiert die weite Verbreitung der funktionalen Programmierung eine Reifung der Softwareentwicklung als Disziplin.
Da unsere Abhängigkeit von verteilten Cloud-Umgebungen, Echtzeit-Datenstreaming und massiv paralleler Verarbeitung kontinuierlich wächst,
wird der historische imperative Ansatz, CPU-Register und veränderlichen Speicher mikrozumanagen, zunehmend unhaltbar. Die funktionale
Programmierung bietet eine höhere Abstraktionsebene, die perfekt auf die architektonischen Anforderungen der modernen Ära abgestimmt ist.
Durch die konsequente Förderung von Unveränderlichkeit, reinen Funktionen, referenzieller Transparenz und starken Typsystemen gibt sie
Entwicklern die konzeptionellen Werkzeuge an die Hand, die notwendig sind, um komplexe Systeme deterministisch zu durchdenken. Obwohl die
anfängliche Lernkurve für diejenigen, die an imperative Modelle gewöhnt sind, steil sein kann, sind die langfristigen Vorteile unbestreitbar:
sauberere Codebasen, drastisch verkürzte Debugging-Zyklen, nahtlose Nebenläufigkeit und von Natur aus zuverlässigere Software. Die funktionale
Programmierung hat den erfolgreichen Übergang von einer esoterischen akademischen Nische zu einer fundamentalen Säule der modernen Technologie
vollzogen und die Art und Weise, wie wir weltweit Software konzipieren, entwerfen und ausführen, permanent verändert."""


@pytest.mark.slow
class SentenceDetectorSaTModelTestSpec(unittest.TestCase):
    def setUp(self):
        self.spark = SparkContextForTest.spark
        self.document_assembler = DocumentAssembler() \
            .setInputCol("text") \
            .setOutputCol("document")

    def _load_model(self):
        return SentenceDetectorSaTModel.pretrained() \
            .setInputCols(["document"]) \
            .setOutputCol("sentence")

    def _segment(self, text, model):
        data = self.spark.createDataFrame([[text]]).toDF("text")
        pipeline = Pipeline(stages=[self.document_assembler, model])
        result = pipeline.fit(data).transform(data)
        result.show(truncate=False)
        return [r.s for r in result.selectExpr("explode(sentence.result) as s").collect()]

    # 1. Serialization round-trip.
    def test_save_and_reload(self):
        save_path = "./tmp_sat_model"
        self._load_model().setThreshold(0.25).write().overwrite().save(save_path)

        reloaded = SentenceDetectorSaTModel.load(save_path) \
            .setInputCols(["document"]) \
            .setOutputCol("sentence")
        self.assertAlmostEqual(reloaded.getThreshold(), 0.25, places=5)

        sentences = self._segment("Hello world. Goodbye world.", reloaded)
        self.assertGreaterEqual(len(sentences), 1)

    # 2. Long English document (exercises multi-window stitching).
    def test_long_english(self):
        self.assertGreater(len(LONG_ENGLISH.split()), 500)
        sentences = self._segment(LONG_ENGLISH, self._load_model())
        self.assertGreaterEqual(len(sentences), 2)

    # 3. Long document in another language (German).
    def test_long_german(self):
        self.assertGreater(len(LONG_GERMAN.split()), 500)
        sentences = self._segment(LONG_GERMAN, self._load_model())
        self.assertGreaterEqual(len(sentences), 2)

    # 4. Length-constrained (Viterbi) segmentation: threshold is ignored and every segment must
    #    respect the character bounds. trimWhitespace is off so lengths are exact.
    def test_min_max_sentence_length(self):
        min_len, max_len = 150, 305
        model = self._load_model() \
            .setMinSentenceLength(min_len) \
            .setMaxSentenceLength(max_len) \
            .setTrimWhitespace(False)

        sentences = self._segment(LONG_ENGLISH, model)
        for i, s in enumerate(sentences):
            print("Sentence {} (length {}): {}".format(i + 1, len(s), s))

        self.assertGreater(len(sentences), 1)
        for s in sentences:
            self.assertGreaterEqual(len(s), min_len)
            self.assertLessEqual(len(s), max_len)
