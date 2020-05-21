package com.read.AI.service;
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.learning.config.Nadam;
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;


//multilayered (MLP) applied to digit classification
//Uses Two input layers and one hidden layer
public class AIService {

    public String getMessage() throws IOException {
        setup();
        return "Hello world";
    }

    private static Logger log = LoggerFactory.getLogger(AIService.class);

    public void setup() {
        //number of rows and columns in the input pictures
        final int numRows = 28;
        final int numColumns = 28;
        int outputNum = 10;// number of output classes
        int batchSize = 64;// batch size for each epoch
        int rngSeed = 123;// random number seed for reproducibility
        int numEpochs = 15;// number of epochs to perform
        double rate = 0.0015;// learning rate




    }
}
