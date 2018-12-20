package com.johnsnowlabs.nlp.annotators.parser.typdep.io;

import com.johnsnowlabs.nlp.annotators.parser.typdep.DependencyInstance;
import com.johnsnowlabs.nlp.annotators.parser.typdep.DependencyPipe;
import com.johnsnowlabs.nlp.annotators.parser.typdep.Options;
import com.johnsnowlabs.nlp.annotators.parser.typdep.util.DependencyLabel;


public class Conll09Writer {

    Options options;
    String[] labels;

    public Conll09Writer(Options options, DependencyPipe pipe) {
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
