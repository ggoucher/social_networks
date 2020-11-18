# social_networks

Terms to know:

    Hidden Layers -- Layers of neurons that do not directly interact with the input or output layers. This allows for 
    more complex functions to process data.
    
    Feedforward -- We are moving information forward in our network. The path is: input -> hidden -> output.
    
    Epochs -- A single step of training.
    
    Backpropogation -- A method of calculating the change in weights where the error is computed between the output 
    values and the correct values. The error value is then sent back through the network in order to help readjust weights.
    
    Alpha -- When calculating changes in weights, we have to prevent them from changing too much per iteration. To do 
    this, we use a small value between 0 and 1 known as the alpha value. This value ensures we do not over correct one 
    way or the other when adjusting weights. The learning rate/alpha value aims to make our network converge on valuable 
    output without taking too much time. 
    
