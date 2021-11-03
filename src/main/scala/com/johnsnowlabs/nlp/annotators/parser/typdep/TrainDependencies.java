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

package com.johnsnowlabs.nlp.annotators.parser.typdep;

import java.io.IOException;
import java.io.Serializable;

public class TrainDependencies implements Serializable {

    private static final long serialVersionUID = 1L;

    private TrainFile trainFile;
    private DependencyPipe dependencyPipe;
    private TypedDependencyParser typedDependencyParser;
    private Options options;
    private Parameters parameters;

    public DependencyPipe getDependencyPipe() {
        return dependencyPipe;
    }

    public void setDependencyPipe(DependencyPipe dependencyPipe) {
        this.dependencyPipe = dependencyPipe;
    }

    public Options getOptions() {
        return options;
    }

    public void setOptions(Options options) {
        this.options = options;
    }

    public Parameters getParameters() {
        return parameters;
    }

    public void setParameters(Parameters parameters) {
        this.parameters = parameters;
    }

    public TrainDependencies(TrainFile trainFile, DependencyPipe dependencyPipe,
                             TypedDependencyParser typedDependencyParser, Options options){
        this.trainFile = trainFile;
        this.dependencyPipe = dependencyPipe;
        this.typedDependencyParser = typedDependencyParser;
        this.options = options;
    }

    public void startTraining() throws IOException {
        DependencyInstance[] dependencyInstances = dependencyPipe.createInstances(trainFile.path(),
                trainFile.conllFormat());
        dependencyPipe.pruneLabel(dependencyInstances);

        typedDependencyParser.setParameters(new Parameters(dependencyPipe, options));
        typedDependencyParser.train(dependencyInstances);

        setOptions(typedDependencyParser.getOptions());
        setParameters(typedDependencyParser.getParameters());
        setDependencyPipe(typedDependencyParser.getDependencyPipe());

    }


}
