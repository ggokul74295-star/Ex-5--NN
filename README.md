<H3>NAME: Gokulan S</H3>
<H3>REGISTER NO.: 212225230078</H3>
<H3>EX. NO.5</H3>
<H3>DATE: 27/08/2026</H3>
<H1 ALIGN =CENTER>Implementation of XOR  using RBF</H1>
<H3>Aim:</H3>
To implement a XOR gate classification using Radial Basis Function  Neural Network.

<H3>Theory:</H3>
<P>Exclusive or is a logical operation that outputs true when the inputs differ.For the XOR gate, the TRUTH table will be as follows XOR truth table </P>

<P>XOR is a classification problem, as it renders binary distinct outputs. If we plot the INPUTS vs OUTPUTS for the XOR gate, as shown in figure below </P>

<P>The graph plots the two inputs corresponding to their output. Visualizing this plot, we can see that it is impossible to separate the different outputs (1 and 0) using a linear equation.
A Radial Basis Function Network (RBFN) is a particular type of neural network. The RBFN approach is more intuitive than MLP. An RBFN performs classification by measuring the input’s similarity to examples from the training set. Each RBFN neuron stores a “prototype”, which is just one of the examples from the training set. When we want to classify a new input, each neuron computes the Euclidean distance between the input and its prototype. Thus, if the input more closely resembles the class A prototypes than the class B prototypes, it is classified as class A ,else class B.
A Neural network with input layer, one hidden layer with Radial Basis function and a single node output layer (as shown in figure below) will be able to classify the binary data according to XOR output.
</P>

<H3>ALGORITHM:</H3>
Step 1: Initialize the input  vector for you bit binary data<Br>
Step 2: Initialize the centers for two hidden neurons in hidden layer<Br>
Step 3: Define the non- linear function for the hidden neurons using Gaussian RBF<br>
Step 4: Initialize the weights for the hidden neuron <br>
Step 5 : Determine the output  function as 
                 Y=W1*φ1 +W1 *φ2 <br>
Step 6: Test the network for accuracy<br>
Step 7: Plot the Input space and Hidden space of RBF NN for XOR classification.

<H3>PROGRAM:</H3>

### Imports and rbf functions:
```python
import numpy as np
import matplotlib.pyplot as plt

def gaussian_rbf(x, landmark, gamma=1.0):
    return np.exp(-gamma * np.sum((np.asarray(x) - np.asarray(landmark))**2))
```

### End to end function with matrix solving:
```python
def end_to_end(X1, X2, ys, mu1, mu2, gamma=1.0):
    X1 = np.asarray(X1)
    X2 = np.asarray(X2)
    ys = np.asarray(ys)
    mu1 = np.asarray(mu1)
    mu2 = np.asarray(mu2)

    points = np.column_stack((X1, X2))
    from_1 = np.array([gaussian_rbf(p, mu1, gamma) for p in points])
    from_2 = np.array([gaussian_rbf(p, mu2, gamma) for p in points])

    plt.figure(figsize=(14, 6))

    plt.subplot(1, 2, 1)
    plt.scatter(X1[ys == 0], X2[ys == 0], color='blue', label='Class 0')
    plt.scatter(X1[ys == 1], X2[ys == 1], color='orange', label='Class 1')
    plt.xlabel('x1', fontsize=14)
    plt.ylabel('x2', fontsize=14)
    plt.title('XOR: Linearly Inseparable', fontsize=16)
    plt.legend()
    plt.grid(alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.scatter(from_1[ys == 0], from_2[ys == 0], color='red', label='Class 0')
    plt.scatter(from_1[ys == 1], from_2[ys == 1], color='green', label='Class 1')
    plt.plot([0, 1], [1, 0], 'k--', linewidth=2)
    plt.annotate('Separating Hyperplane', xy=(0.4, 0.55), xytext=(0.55, 0.75),
                 arrowprops=dict(facecolor='black', shrink=0.05), fontsize=12)
    plt.xlabel(f'RBF to mu1: {mu1}', fontsize=14)
    plt.ylabel(f'RBF to mu2: {mu2}', fontsize=14)
    plt.title('Transformed Inputs: Linearly Separable', fontsize=16)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

    design_matrix = np.column_stack((from_1, from_2, np.ones_like(from_1)))
    weights = np.linalg.lstsq(design_matrix, ys, rcond=None)[0]

    predictions = np.round(design_matrix.dot(weights))
    print('Predicted labels:', predictions.astype(int))
    print('True labels     :', ys.astype(int))
    print(f'Weights: {weights}')
    return weights
```

### Prediction function:
```python
def predict_matrix(point, weights, mu1, mu2, gamma=1.0):
    point = np.asarray(point)
    rbf_1 = gaussian_rbf(point, mu1, gamma)
    rbf_2 = gaussian_rbf(point, mu2, gamma)
    feature_vector = np.array([rbf_1, rbf_2, 1.0])
    return np.round(feature_vector.dot(weights)).astype(int)
```

### Data setup and Execution:
```python
x1 = np.array([0, 0, 1, 1])
x2 = np.array([0, 1, 0, 1])
ys = np.array([0, 1, 1, 0])
mu1 = np.array([0, 1])
mu2 = np.array([1, 0])

trained_weights = end_to_end(x1, x2, ys, mu1, mu2, gamma=1.0)
```

### Testing with predictions:
```python
test_points = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])

for point in test_points:
    pred = predict_matrix(point, trained_weights, mu1, mu2, gamma=1.0)
    print(f'Input: {point}, Predicted: {pred}')
```

<H3>OUTPUT:</H3>

### Transformed inputs:
<img width="1389" height="590" alt="image" src="https://github.com/user-attachments/assets/fe227733-3c75-4e17-b561-0526189addaf" />

```ipynb
Predicted labels: [0 1 1 0]
True labels     : [0 1 1 0]
Weights: [ 2.5026503   2.5026503  -1.84134719]
```

### Testing with predictions:
```ipynb
Input: [0 0], Predicted: 0
Input: [0 1], Predicted: 1
Input: [1 0], Predicted: 1
Input: [1 1], Predicted: 0
```

<H3>Result:</H3>
Thus, a Radial Basis Function Neural Network is implemented to classify XOR data.








