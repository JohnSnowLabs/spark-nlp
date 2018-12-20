package com.johnsnowlabs.nlp.annotators.parser.typdep;

import java.io.Serializable;

public class Options implements Serializable {

    private static final long serialVersionUID = 1L;

    String unimapFile = null;

    int numberOfPreTrainingIterations = 2;
    private int numberOfTrainingIterations;
    boolean initTensorWithPretrain = true;
    float regularization = 0.01f;
    float gammaLabel = 0;
    int rankFirstOrderTensor = 50;
    int rankSecondOrderTensor = 30;

    public int getNumberOfTrainingIterations() {
        return numberOfTrainingIterations;
    }

    public void setNumberOfTrainingIterations(int numberOfTrainingIterations) {
        this.numberOfTrainingIterations = numberOfTrainingIterations;
    }

    public Options() {

    }

    private Options(Options options) {
        this.numberOfPreTrainingIterations = options.numberOfPreTrainingIterations;
        this.numberOfTrainingIterations = options.numberOfTrainingIterations;
        this.initTensorWithPretrain = options.initTensorWithPretrain;
        this.regularization = options.regularization;
        this.gammaLabel = options.gammaLabel;
        this.rankFirstOrderTensor = options.rankFirstOrderTensor;
        this.rankSecondOrderTensor = options.rankSecondOrderTensor;
    }

    static Options newInstance(Options options){
        //Copy factory
        return new Options(options);
    }

}
