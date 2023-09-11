package utils;

public class SGD implements Optimizer {
	
	public final double lr;
	public final double momentum;
	
	public SGD(double lr, double momentum) {
		this.lr = lr;
		this.momentum = momentum;
	}
	
}
