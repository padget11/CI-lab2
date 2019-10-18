import numpy
import matplotlib.pyplot as plt

# A single perceptron function
def perceptron(inputs_list, weights_list, bias):
    # Convert the inputs list into a numpy array
    inputs = numpy.array(inputs_list)

    # Convert the weights list into a numpy array
    weights = numpy.array(weights_list)

    # Calculate the dot product
    summed = numpy.dot(inputs, weights)

    # Add in the bias
    summed = summed + bias

    # Calculate output
    # N.B this is a ternary operator, neat huh?
    output = 1 if summed > 0 else 0

    return output

# binary inputs
inputs_0 = [0.0, 0.0]
inputs_1 = [0.0, 1.0]
inputs_2 = [1.0, 0.0]
inputs_3 = [1.0, 1.0]
inputs = [inputs_0, inputs_1, inputs_2, inputs_3]

# outputs for each gate, initialise to 0
outputs_AND = [0, 0, 0, 0]
outputs_OR = [0, 0, 0, 0]
outputs_NAND = [0, 0, 0, 0]
outputs_NOR = [0, 0, 0, 0]
outputs_XOR = [0, 1, 1, 0]
outputs = [outputs_AND, outputs_OR, outputs_NAND, outputs_NOR]

# weights for gates
weights_AND = [1.0, 1.0]
weights_OR = [1.0, 1.0]
weights_NAND = [-1.0, -1.0]
weights_NOR = [-1.0, -1.0]
weights = [weights_AND, weights_OR, weights_NAND, weights_NOR]

# bias for gates
# AND -2 < b <= -1
# OR -1 < b <= 0
# NAND 1 < b <= 2
# NOR 0 < b <= 1
bias = [-1.5, -0.5, 1.5, 0.5]

titles = ["AND", "OR", "NAND", "NOR"]

# For each gate calculate the outputs of the preceptron and plot on graph
for x in range(len(outputs)):
    plt.subplot(2, 2, x + 1)
    for i in range(len(inputs)):
        outputs[x][i] = perceptron(inputs[i], weights[x], bias[x])
        if outputs[x][i] == 1:
            # plot green
            plt.scatter(inputs[i][0], inputs[i][1], s=50, color="green", zorder=3)
        else:
            # plot red
            plt.scatter(inputs[i][0], inputs[i][1], s=50, color="red", zorder=3)

    # Plot line
    x1 = numpy.linspace(-2,2,100)
    x2 = -1*(weights[x][0]/weights[x][1])*x1 - (bias[x]/weights[x][1])
    plt.plot(x1, x2, '-b', label='x1 = (-w1/w2) * x1 - (b/w2)')
    plt.title(titles[x])
    plt.legend()

    # Set the axis limits
    plt.xlim(-1, 2)
    plt.ylim(-1, 2)

    # Label the plot
    plt.xlabel("Input 1")
    plt.ylabel("Input 2")
   
    # Turn on grid lines
    plt.grid(True, linewidth=1, linestyle=':')

    # Autosize (stops the labels getting cut off)
    plt.tight_layout()


# Show the plot
plt.show()



