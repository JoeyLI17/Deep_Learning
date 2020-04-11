import numpy as np


class NeuralNetwork(object):
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        # Set number of nodes in input, hidden and output layers.
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes
        
        # print("Check input_nodes \n" , self.input_nodes)
        # print("Check hidden_nodes \n" , self.hidden_nodes)
        # print("Check output_nodes \n" , self.output_nodes)
        # print("Check learning_rate \n" , self.learning_rate)

        # Initialize weights
        self.weights_input_to_hidden = np.random.normal(0.0, self.input_nodes**-0.5, 
                                       (self.input_nodes, self.hidden_nodes))

        self.weights_hidden_to_output = np.random.normal(0.0, self.hidden_nodes**-0.5, 
                                       (self.hidden_nodes, self.output_nodes))
        self.lr = learning_rate
        
        # print("Check weights_input_to_hidden \n" , self.weights_input_to_hidden)
        # print("Check weights_hidden_to_output \n" , self.weights_hidden_to_output)
        # print("\n")
        
        #### TODO: Set self.activation_function to your implemented sigmoid function ####
        #
        # Note: in Python, you can define a function with a lambda expression,
        # as shown below.
        self.activation_function = lambda x : 1 / (1 + np.exp(-x))  # Replace 0 with your sigmoid calculation.
        
        ### If the lambda code above is not something you're familiar with,
        # You can uncomment out the following three lines and put your 
        # implementation there instead.
        #
        #def sigmoid(x):
        #    return 0  # Replace 0 with your sigmoid calculation here
        #self.activation_function = sigmoid
                    

    def train(self, features, targets):
        ''' Train the network on batch of features and targets. 
        
            Arguments
            ---------
            
            features: 2D array, each row is one data record, each column is a feature
            targets: 1D array of target values
        
        '''
        n_records = features.shape[0]
        delta_weights_i_h = np.zeros(self.weights_input_to_hidden.shape)
        delta_weights_h_o = np.zeros(self.weights_hidden_to_output.shape)
        # print("debug train: delta_weights_h_o shape: ", delta_weights_h_o.shape) # 2x1
        for X, y in zip(features, targets):
            
            final_outputs, hidden_outputs = self.forward_pass_train(X)  # Implement the forward pass function below
            # Implement the backproagation function below
            delta_weights_i_h, delta_weights_h_o = self.backpropagation(final_outputs, hidden_outputs, X, y, 
                                                                        delta_weights_i_h, delta_weights_h_o)
        self.update_weights(delta_weights_i_h, delta_weights_h_o, n_records)


    def forward_pass_train(self, X):
        ''' Implement forward pass here 
         
            Arguments
            ---------
            X: features batch

        '''
        # print("debug forward pass train: input check: \n",X,X.shape) # 1x3
        # print("debug forward pass train: weights_input_to_hidden check: \n ",self.weights_input_to_hidden,self.weights_input_to_hidden.shape) # 3x2
        # print("debug forward pass train: weights_hidden_to_output check: \n ",self.weights_hidden_to_output,self.weights_hidden_to_output.shape) # 2x1
        #### Implement the forward pass here ####
        ### Forward pass ###
        # TODO: Hidden layer - Replace these values with your calculations.
        hidden_inputs = np.dot(X , self.weights_input_to_hidden) # signals into hidden layer
        hidden_outputs = self.activation_function(hidden_inputs) # signals from hidden layer
        # print("debug forward pass train: hidden_inputs check: \n ",hidden_inputs,hidden_inputs.shape) # 1x2 X_h OK
        # print("debug forward pass train: hidden_outputs check: \n ",hidden_outputs,hidden_outputs.shape) # 1x2 X_h OK

        # TODO: Output layer - Replace these values with your calculations.
        final_inputs = np.dot(hidden_outputs , self.weights_hidden_to_output) # signals into final output layer # 1x2 2x1 1x1
        final_outputs = final_inputs # This is a regression problem 
        # print("debug forward pass train: final_inputs check: \n ",final_inputs,final_inputs.shape) # 1x1 OK
        
        return final_outputs, hidden_outputs

    def backpropagation(self, final_outputs, hidden_outputs, X, y, delta_weights_i_h, delta_weights_h_o):
        # print("debug backpropagation: final_outputs shape: ", final_outputs) # 2x1
        # print("debug backpropagation: hidden_outputs shape: ", hidden_outputs) # 2x1
        # print("debug backpropagation: X shape: ", X) # 2x1
        # print("debug backpropagation: y shape: ", y) # 2x1
        # print("debug backpropagation: delta_weights_i_h: ", delta_weights_i_h) # 2x1
        # print("debug backpropagation: delta_weights_h_o : ", delta_weights_h_o) # 2x1
        # print("debug backpropagation: weights_input_to_hidden  : ", self.weights_input_to_hidden ) # 2x1
        # print("debug backpropagation: weights_hidden_to_output  : ", self.weights_hidden_to_output ) # 2x1
        # print("debug backpropagation input check done.\n")
        ''' Implement backpropagation
         
            Arguments
            ---------
            final_outputs: output from forward pass
            y: target (i.e. label) batch
            delta_weights_i_h: change in weights from input to hidden layers
            delta_weights_h_o: change in weights from hidden to output layers

        '''
        #### Implement the backward pass here ####
        ### Backward pass ###

        # TODO: Output error - Replace this value with your calculations.
        error = y - final_outputs # Output layer error is the difference between desired target and actual output.
        # print("debug backpropagation: error \n" , (error), error.shape) # 1x1 OK

        # TODO: Backpropagated error terms - Replace these values with your calculations.
        output_error_term = error # This is a regression problem 
        # print("debug backpropagation: output_error_term \n" , (output_error_term),output_error_term.shape) # 1x1 OK
         
        # TODO: Calculate the hidden layer's contribution to the error        
        hidden_error = np.dot(self.weights_hidden_to_output, output_error_term) # f(Wa) 2x1 1x1 2x1
        # print("debug backpropagation: hidden_error \n" , (hidden_error), hidden_error.shape) # OK
        
        hidden_error_term = hidden_error * (hidden_outputs * (1 - hidden_outputs)) # 2x1 1x2 2x2
        # print("debug backpropagation: hidden_outputs * (1 - hidden_outputs) \n" , (hidden_outputs * (1 - hidden_outputs)), (hidden_outputs * (1 - hidden_outputs)).shape) # 1x2
        # print("debug backpropagation: hidden_error_term \n" , (hidden_error_term), hidden_error_term.shape) # 1x2 OK
        
        # Weight step (hidden to output)
        # delta_weights_h_o += output_error_term * hidden_outputs
        temp = output_error_term * hidden_outputs
        temp_2 = temp[:,None]
        # print("debug backpropagation: hidden_outputs * output_error_term \n" , (temp), (temp).shape) # 1x2
        # print("debug backpropagation: temp_2 \n" , (temp_2), (temp_2).shape) # 1x2
        delta_weights_h_o += temp_2 
        # print("debug backpropagation: delta_weights_h_o \n" , (delta_weights_h_o), delta_weights_h_o.shape)
        
        # Weight step (input to hidden)
        delta_weights_i_h += hidden_error_term * X[:, None]
        # print("debug backpropagation: delta_weights_i_h \n" , (delta_weights_i_h), delta_weights_i_h.shape) # OK
        return delta_weights_i_h, delta_weights_h_o

    def update_weights(self, delta_weights_i_h, delta_weights_h_o, n_records):
        ''' Update weights on gradient descent step
         
            Arguments
            ---------
            delta_weights_i_h: change in weights from input to hidden layers
            delta_weights_h_o: change in weights from hidden to output layers
            n_records: number of records

        '''
        self.weights_hidden_to_output += self.lr * delta_weights_h_o / n_records # update hidden-to-output weights with gradient descent step
        self.weights_input_to_hidden += self.lr * delta_weights_i_h / n_records # update input-to-hidden weights with gradient descent step
        # print("debug weights_hidden_to_output: \n",self.weights_hidden_to_output)
        # print("debug weights_input_to_hidden: \n",self.weights_input_to_hidden)

    def run(self, features):
        ''' Run a forward pass through the network with input features 
        
            Arguments
            ---------
            features: 1D array of feature values
        '''
        # print("debug run: features \n", features, features.shape)
        # print("debug run: weights_input_to_hidden check: \n ",self.weights_input_to_hidden,self.weights_input_to_hidden.shape) # 3x2
        # print("debug run: weights_hidden_to_output check: \n ",self.weights_hidden_to_output,self.weights_hidden_to_output.shape) # 2x1
        #### Implement the forward pass here ####
        # TODO: Hidden layer - replace these values with the appropriate calculations.
        hidden_inputs = np.dot(features , self.weights_input_to_hidden) # signals into hidden layer
        hidden_outputs = self.activation_function(hidden_inputs) # signals from hidden layer, new feature
        # print("debug run: hidden_inputs \n", hidden_inputs)
        # print("debug run: hidden_outputs \n", hidden_outputs)
        
        # TODO: Output layer - Replace these values with the appropriate calculations.
        final_inputs = np.dot(hidden_outputs, self.weights_hidden_to_output) # signals into final output layer
        # print("debug run: final_inputs \n", final_inputs)
        # print("debug run: done")
        
        return final_inputs


#########################################################
# Set your hyperparameters here
##########################################################
iterations = 3000
learning_rate = 0.7
hidden_nodes = 10
output_nodes = 1
