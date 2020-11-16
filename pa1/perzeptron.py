import random
import math

# Evaluation function that measures the performance of the perzeptron by comparing the predictions to true values

def evaluate(Y, N, M, w_nm, train_len):
    avg_error = 0

    for i in range(0, train_len):

        # calculation of weighted sum, net_m

        net_m = [0] * M

        for j in range(0, M):
            net_m[j] = w_nm[N][j]  # initialize with bias weight
            for k in range(0, N):
                net_m[j] += w_nm[k][j] * X[i][k]

        for j in range(0, M):
            avg_error += (Y[i][j] - net_m[j]) ** 2  # distance measure for performance

    print(math.sqrt(avg_error))


# Predicts the output given the input by using the weights

def predict(x, N, M, w_nm):
    out = [0] * M
    net_m = [0] * M
    for j in range(0, M):
        net_m[j] = w_nm[N][j];  # initialize with bias weight
        for k in range(0, N):
            net_m[j] += w_nm[k][j] * x[k]

        out[j] = int(net_m[j] > 0)
    return out


# Predicts all of the input against all of the output and prints accuracy

def test(X, Y, w_nm):
    N = len(X[0])
    M = len(Y[0])
    result = [predict(x, N, M, w_nm) == y for x, y in zip(X, Y)]
    print("accuracy ", sum(result) / len(result))


# Trains the perceptron with a learning rate and epoch count that specifies how many times it will go over the same training data

def train(X, Y, w_nm, learning_rate, epoch_count):
    N = len(X[0])
    M = len(Y[0])
    train_len = len(X)

    w_nm = []

    # Initialize the weights randomly
    for i in range(0, N + 1):
        w_m = []
        for j in range(0, M):
            w_m.append(random.uniform(-0.5, 0.5))

        w_nm.append(w_m)

    for e in range(0, epoch_count):

        # another loop for training examples
        for i in range(0, train_len):

            net_m = [0] * M
            error = [0] * M

            # calculation of weighted sum, net_m
            for j in range(0, M):
                net_m[j] = w_nm[N][j];  # initialize with bias weight
                for k in range(0, N):
                    net_m[j] += w_nm[k][j] * X[i][k]

            # weight updates according to widrow-hoff learning rule
            for j in range(0, M):
                error[j] = Y[i][j] - net_m[j]
                w_nm[N][j] += learning_rate * (error[j])  # bias update
                for k in range(0, N):
                    w_nm[k][j] += learning_rate * error[j] * X[i][k]

        if DEBUG:
            evaluate(Y, N, M, w_nm, train_len)  # to see the performance

    return w_nm


# Saves the weights to a file named "weights.txt"

def save_model(w_nm, N, M):
    par = "# Weights: rows for output and columns for input. Each line has the weights for connections of one output neuron to the input neurons. \n"
    for i in range(0, M):
        for j in range(0, N):
            par += str(w_nm[j][i]) + "\t"
        par += "\n"

    par += "\n"
    par += "# Weights for bias for the output neurons. Each value in the line is the weight of the corresponding output neuron connected to the BIAS. \n"
    for i in range(0, M):
        par += str(w_nm[N][i]) + "\t"
    par += "\n"
    with open("weights.txt", "w+") as f:
        f.write(par)

if __name__ == '__main__':
    DEBUG = 0
    # Variables

    X = []
    Y = []
    w_nm = []

    # Getting data from the text file and saving them to lists
    with open("PA-A-train.txt") as f:
        for line in f:
            if line[0] != "#":
                tab_separated = line.strip().split("\t")
                x_str = tab_separated[0].strip()
                y_str = tab_separated[1]
                X.append([int(x) for x in x_str.split(" ")])
                Y.append([int(y) for y in y_str.split(" ")])

    w_nm = train(X, Y, w_nm, 0.01, 100)
    save_model(w_nm, len(X[0]), len(Y[0]))
