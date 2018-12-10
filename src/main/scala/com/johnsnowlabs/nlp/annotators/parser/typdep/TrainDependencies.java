package com.johnsnowlabs.nlp.annotators.parser.typdep;

import java.io.IOException;
import java.io.Serializable;

public class TrainDependencies implements Serializable {

    private static final long serialVersionUID = 1L;

    private String trainFile;
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

    public TrainDependencies(String trainFile, DependencyPipe dependencyPipe,
                             TypedDependencyParser typedDependencyParser, Options options){
        this.trainFile = trainFile;
        this.dependencyPipe = dependencyPipe;
        this.typedDependencyParser = typedDependencyParser;
        this.options = options;
    }

    public void startTraining() throws IOException {
        DependencyInstance[] dependencyInstances = dependencyPipe.createInstances(trainFile);
        dependencyPipe.pruneLabel(dependencyInstances);

        typedDependencyParser.setParameters(new Parameters(dependencyPipe, options));
        typedDependencyParser.train(dependencyInstances);

        setOptions(typedDependencyParser.getOptions());
        setParameters(typedDependencyParser.getParameters());
        setDependencyPipe(typedDependencyParser.getDependencyPipe());

    }


}
