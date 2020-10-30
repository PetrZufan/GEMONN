## Introduction
This the source code of GEMONN *A Gradient Guided Evolutionary Approach to Training Deep Neural Networks*

## Requirements and  Dependency
```bash
torch>=1.4.0
torchvision >=0.5.1
cuda 10.0
cupy
numpy
matplotlib
```
## Usage
```bash
python Main.py
python Main.py --Generations 500 --Popsize 50 --HiddenNum 500 --plot  --save --save_dir ./result
```
## Result 
This is the results of Auto-Encoder (AE) with sinlge hidden layer on MNIST results:
the x-axis is MSE loss, and the y-axis is the sparsity in term of L0-norm

Initial result: 
![Solution of initialization SSGD](./result/plot_initialization.png)

Result after evolution: 
![Solution before SSGD](./result/plot_499.png)

Final results:
![Solution after SSGD](./result/plot_500.png)

Next, the comparison of GEMONN, SGD, Adam, and NSGA-II vatiants are shown as follows:
![MSE_pareto](./result/MSE_Pareto.eps)

## Extension on more models
For training more models such as LSTM and CNNs, there are guidance  in  `Private_function.py' :

1) Get your model in pytorch
```bash
Model = LeNet()
```
2) Get the weights dictionary of model :
```bash
 Parameter_dict = Model.state_dict()
```
3) Initialize the population and obtain corresponding size and length inforamtion of weighs in different parts of the model :
```bash
 Population, Boundary, Coding, SizeInform, LengthInform = Initialization_Pop_(PopSize =10, Model = Model)
```
4) Obtain the weights dictionary of each individual in population and compte the inference loss for evaluation:
```bash
 Parameter_dict_i = Pop2weights(Population[0], SizeInform, LengthInform, Parameter_dict)
 Model.load_state_dict(Parameter_dict_i)
```
5) Train this model by GEMONN supported by sparse-SGD or spare-Adam



