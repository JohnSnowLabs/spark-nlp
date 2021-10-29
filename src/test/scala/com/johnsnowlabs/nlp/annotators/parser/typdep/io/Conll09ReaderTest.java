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

package com.johnsnowlabs.nlp.annotators.parser.typdep.io;

import com.johnsnowlabs.nlp.annotators.parser.typdep.ConllData;
import com.johnsnowlabs.nlp.annotators.parser.typdep.DependencyInstance;
import org.junit.Test;

import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertNull;

public class Conll09ReaderTest {

    @Test
    public void shouldCreateDependencyInstanceWhenSentenceWithConll09FormatIsSent() {

        Conll09Reader conll09Reader = new Conll09Reader();
        ConllData[] sentence = new ConllData[2];
        ConllData wordForms = new ConllData("The", "the","DT", "DT","_", 2, 0, 2);
        sentence[0] = wordForms;
        wordForms = new ConllData("economy", "economy","NN", "NN", "_", 4, 4,10);
        sentence[1] = wordForms;

        DependencyInstance dependencyInstance = conll09Reader.nextSentence(sentence);

        assertNotNull("dependencyInstance should not be null", dependencyInstance);
    }

    @Test
    public void shouldReturnNullWhenNoSentenceISent(){
        Conll09Reader conll09Reader = new Conll09Reader();
        ConllData[] sentence = new ConllData[1];
        ConllData wordForms = new ConllData("end", "sentence","ES", "ES","ES", -2, 0, 0);
        sentence[0] = wordForms;
        DependencyInstance dependencyInstance = conll09Reader.nextSentence(sentence);

        assertNull("dependencyInstance should be null", dependencyInstance);
    }

}