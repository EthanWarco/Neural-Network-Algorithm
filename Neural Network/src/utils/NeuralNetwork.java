package utils;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.concurrent.ThreadLocalRandom;

public class NeuralNetwork {
	
	private final List<Layer> layers = new ArrayList<Layer>();
	private final double[][] trainingData;
	private final double[][] labels;
	private final double[][] testvals;
	private final double[][] testlabels;
	private final int batchSize;
	private final Optimizer optimizer;
	
	public NeuralNetwork(double[][] trainingData, double[][] labels, double[][] testvals, double[][] testlabels, Optimizer optimizer, int batchSize) {
		this.labels = labels;
		this.trainingData = trainingData;
		this.testlabels = testlabels;
		this.testvals = testvals;
		this.optimizer = optimizer;
		this.batchSize = batchSize;
	}
	
	public void addLayer(Layer layer) {
		layers.add(layer);
		int index = layers.indexOf(layer);
		if(index > 0) layers.get(index-1).fullyConnect(layer);
	}
	
	private void resetDerivatives() {
		for(int i = 0; i < layers.size(); i++) {
			for(int j = 0; j < layers.get(i).neurons.length; j++) {
				layers.get(i).neurons[j].dcost_dout = 0.0;
			}
		}
	}
	
	
	
	public void train(int epoch, int intervalBetweenUpdates) {
		for(int i = 0; i < epoch; i++) {
			shuffleTrainingData();
			double sum = 0;
			for(int j = 0; j < trainingData.length; j++) {
				feedForward(trainingData[j]);
				backPropagate(labels[j]);
				resetDerivatives();
				if(j%batchSize == 0) updateParams();
				for(int k = 0; k < labels[j].length; k++) sum += -labels[j][k] * Math.log(layers.get(layers.size()-1).neurons[k].value);
			}
			//Includes both testing and training loss so you know if it is being overfitted
			if(i%intervalBetweenUpdates == 0) {
				System.out.println(i+1 + "/" + epoch + " Training Loss: " + sum/trainingData.length);
				System.out.println(i+1 + "/" + epoch + " Testing Loss: " + getTestingLoss());
				System.out.println();
			}
		}
	}
	
	private void shuffleTrainingData() {
		Random rand = ThreadLocalRandom.current();
		for(int i = trainingData.length-1; i > 0; i--) {
			int index = rand.nextInt(i+1);
			
			double[] data = trainingData[index];
			double[] label = labels[index];
			
			trainingData[i] = data;
			labels[i] = label;
		}
	}
	
	public double getTestingLoss() {
		double sum = 0;
		for(int i = 0; i < testvals.length; i++) {
			feedForward(testvals[i]);
			for(int j = 0; j < testlabels[i].length; j++) sum += -testlabels[i][j] * Math.log(layers.get(layers.size()-1).neurons[j].value);
		}
		return sum/testvals.length;
	}
	
	public double[] predict(double[] input) {
		feedForward(input);
		double[] results = new double[layers.get(layers.size()-1).neurons.length];
		for(int i = 0; i < results.length; i++) results[i] = layers.get(layers.size()-1).neurons[i].value;
		return results;
	}
	
	private void feedForward(double[] input) {
		for(int i = 0; i < input.length; i++) {
			layers.get(0).neurons[i].value = input[i];
		}
		
		for(int i = 1; i < layers.size(); i++) {
			Layer inputLayer = layers.get(i-1);
			Layer outputLayer = layers.get(i);
			for(int j = 0; j < outputLayer.neurons.length; j++) {
				double dotProduct = 0.0;
				for(int k = 0; k < inputLayer.neurons.length; k++) {
					dotProduct += inputLayer.neurons[k].value * inputLayer.neurons[k].connections.get(outputLayer.neurons[j]);
				}
				outputLayer.neurons[j].value = dotProduct + outputLayer.neurons[j].bias;
			}
			outputLayer.activate();
		}
	}
	
	private void backPropagate(double[] actualValues) {
		for(int i = 1; i < layers.size(); i++) {
			layers.get(layers.size()-i-1).backPropagate(actualValues);
		}
	}
	
	private void updateParams() {
		for(int i = 1; i < layers.size(); i++) {
			layers.get(layers.size()-i-1).updateParams(optimizer, batchSize);
		}
	}
	
	public void test() {
		System.out.println("bruh");
		Layer output = layers.get(layers.size()-1);
		int[] correctPredictions = new int[output.neurons.length];
		int[] falsePredictions = new int[output.neurons.length];
		int[] appearances = new int[output.neurons.length];
		for(int i = 0; i < output.neurons.length; i++) {
			correctPredictions[i] = 0;
			falsePredictions[i] = 0;
			appearances[i] = 0;
		}
		
		for(int i = 0; i < testvals.length; i++) {
			feedForward(testvals[i]);
			int label = 0;
			for(int j = 0; j < testlabels[i].length; j++) if(testlabels[i][j] == 1) label = j;
			appearances[label]++;
			int high = 0;
			for(int j = 0; j < output.neurons.length; j++) {
				if(output.neurons[j].value > output.neurons[high].value) {
					high = j;
				}
			}
			if(high == label) correctPredictions[label]++;
			else falsePredictions[high]++;
		}
		
		for(int i = 0; i < output.neurons.length; i++) {
			System.out.println("Input: " + i);
			System.out.println("Correct Predictions: " + correctPredictions[i]);
			System.out.println("False Predictions: " + falsePredictions[i]);
			System.out.println("Accuracy Rate: " + ((double)correctPredictions[i]/(double)appearances[i])*100.0 + "%");
			System.out.println();
		}
	}
	
}
