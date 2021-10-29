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

import com.johnsnowlabs.nlp.annotators.parser.typdep.DependencyInstance;
import com.johnsnowlabs.nlp.annotators.parser.typdep.DependencyPipe;
import com.johnsnowlabs.nlp.annotators.parser.typdep.Options;
import com.johnsnowlabs.nlp.annotators.parser.typdep.util.DependencyLabel;


public class ConllWriter {

    Options options;
    String[] labels;

    public ConllWriter(Options options, DependencyPipe pipe) {
        this.options = options;
        this.labels = pipe.getTypes();
    }

    public DependencyLabel[] getDependencyLabels(DependencyInstance dependencyInstance,
                                                 int[] predictedHeads, int[] predictedLabels){
        int lengthSentence = dependencyInstance.getLength();
        DependencyLabel[] dependencyLabels = new DependencyLabel[lengthSentence];
        for (int i = 1, N = lengthSentence; i < N; ++i) {

            String token = dependencyInstance.getForms()[i];
            String label = labels[predictedLabels[i]];
            int head = predictedHeads[i];
            int start = dependencyInstance.getBegins()[i];
            int end = dependencyInstance.getEnds()[i];

            DependencyLabel dependencyLabel = new DependencyLabel(token, label, head, start, end);
            dependencyLabels[i-1] = dependencyLabel;
        }
        return dependencyLabels;
    }

}
