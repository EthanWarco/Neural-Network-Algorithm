package main;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.Scanner;

import utils.SGD;
import utils.Layer;
import utils.NeuralNetwork;
import utils.Optimizer;

public class Main {
	
	private static final double[][] trainingData = new double[60000][784];
	private static final double[][] trainingLabels = new double[60000][10];
	private static final double[][] testData = new double[10000][784];
	private static final double[][] testLabels = new double[10000][10];
	
	public static void main(String[] args) throws NumberFormatException, IOException {
		setData();
		
		// Sample neural network model - may take hours to finish training
		// First layer has to be 784 neurons and no biases
		Optimizer optimizer = new SGD(0.01, 0.9);
		NeuralNetwork nn = new NeuralNetwork(trainingData, trainingLabels, testData, testLabels, optimizer, 16);
		nn.addLayer(new Layer(784, "input", false));
		nn.addLayer(new Layer(200, Layer.RELU, true));
		nn.addLayer(new Layer(200, Layer.RELU, true));
		nn.addLayer(new Layer(10, Layer.SOFTMAX, true));
		nn.train(10, 1);
		nn.test();
		
	}
	
	private static void setData() {
		File train = new File("C:\\Users\\user\\Desktop\\MNIST Database\\mnist_train.csv");
		File test = new File("C:\\Users\\user\\Desktop\\MNIST Database\\mnist_test.csv");
		try {
			parseCSVFiles(train, test);
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		}
		System.out.println("Parsing Done.");
		System.out.println();
	}
	
	// Parses the information in the two csv files and converts them into the training data and test data
	private static void parseCSVFiles(File train, File test) throws FileNotFoundException {
		Scanner inputStream1 = new Scanner(train);
		int i = 0;
		while(inputStream1.hasNext()) {
			String input = inputStream1.next();
			String[] vals = input.split(",");

			int actualValue = Integer.parseInt(vals[0]);
			trainingLabels[i][actualValue] = 1.0;
			for(int j = 0; j < 10; j++) if(j !=  actualValue) trainingLabels[i][j] = 0.0;
			
			for(int j = 0; j < 784; j++) trainingData[i][j] = ((double)Integer.parseInt(vals[j+1]))/255.0;
			i++;
		}
		inputStream1.close();

		Scanner inputStream2 = new Scanner(test);
		i = 0;
		while(inputStream2.hasNext()) {
			String input = inputStream2.next();
			String[] vals = input.split(",");
			
			testLabels[i] = new double[10];
			int val = Integer.parseInt(vals[0]);
			testLabels[i][val] = 1;
			for(int j = 0; j < 10; j++) if(j != val) testLabels[i][j] = 0;
			for(int j = 0; j < 784; j++) testData[i][j] = ((double)Integer.parseInt(vals[j+1]))/255.0;
			i++;
		}
		inputStream2.close();
	}
	
}