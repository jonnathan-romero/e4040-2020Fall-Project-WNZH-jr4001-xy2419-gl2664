import tensorflow as tf
from tensorflow.keras.layers import Input,Dense
from tensorflow.keras import Model
import sklearn.model_selection as ms
tf.keras.backend.set_floatx('float64')

#  Reference from the code of assignment2 - LeNet_train.py and model_LeNet.py
#  which locates under the directory: /e4040-2020fall-assign2/main/utils/neuralnets/cnn
#  learn how to custom the layers and Models

class Encoder(tf.keras.layers.Layer):
    
    """
    Args:
     - hidden_nodes: The desirable Encoded dimension you want to generate
     - sparsity: On average, for each individual node in the encoded layer, what percentage of the training sample 
                 you want them to be activated through training.
     - regu: The strength of the sparsity regularization
    """
    
    def __init__(self, hidden_nodes, sparsity,regu, **kwargs):
        super().__init__(**kwargs)
        self.units=hidden_nodes
        self.sparsity = sparsity
        self.strength = regu
        
    # Use the build function can make the layer feed into any dimension of the input data without knowing it beforehand.
    # The default encoded layer is stacked by two Dense layers
    # which are delicatedly initialized and use fine-tuned activation function - LeakyReLU
    
    def build(self,input_shape):
        
        self.encoded_layer1=Dense(64, 
                                  # Make the 64 dimension as default
                     activation=tf.keras.layers.LeakyReLU(alpha=0.3),
                                  # By trial and error, alpha=0.3 is optimal 
                     kernel_initializer = tf.keras.initializers.RandomNormal(mean=0., stddev=0.05))
        
        self.encoded_layer2 = Dense(self.units,
                           activation=tf.keras.layers.LeakyReLU(alpha=0.3),
                           # Leaky RuLU causes smoother convergence, though ReLU generates better sparsity property
                           kernel_initializer = tf.keras.initializers.RandomNormal(mean=0., stddev=0.05),
                           activity_regularizer=self.kl_divergence_loss)
    
    # Use the KL divergence loss function as the way to calculate the sparsity regularization term
    def kl_divergence_loss(self,x):
              
        mean = tf.reduce_mean(x,axis=0) # get the mean activated value across each batch
        
        kld=tf.keras.losses.KLDivergence()
        # The strength comes into play directly in the KLD loss
        kld_losses = self.strength*(kld(self.sparsity, mean) + kld(1-self.sparsity, 1-mean))
        
        return kld_losses
        
    def call(self,inputs):
        
        h=self.encoded_layer1(inputs)
        return self.encoded_layer2(h)
    
    
class Decoder(tf.keras.layers.Layer):
    
    """
    Args:
     - reconstruction_nodes: The number of nodes of the output layer 
                             (In most case, should be consistent with the input dimension of the encoded layer)
    """
    
    def __init__(self, reconstruction_nodes, **kwargs):
        super().__init__(**kwargs)
        self.units=reconstruction_nodes
        
    def build(self, input_shape):
        self.decoded_layer1=Dense(64, 
                     activation=tf.keras.layers.LeakyReLU(alpha=0.3),
                     kernel_initializer = tf.keras.initializers.RandomNormal(mean=0., stddev=0.05))

        self.decoded_layer2=Dense(self.units, 
                     activation='linear', # could add non-linear activation function, but here linear turns out best
                     kernel_initializer = tf.keras.initializers.RandomNormal(mean=0., stddev=0.05))
        #self.layer = Dense(self.units, activation = 'linear')
        
    def call(self,inputs):
        
        h=self.decoded_layer1(inputs)
        return self.decoded_layer2(h)
    
# After cumstomize both the encoder and decoder layer
# We define a customized Model as the model_LeNet in assignment 2
# which walks through the customized encoder and decoder layer sequentially
class AE(Model):
    
    """
    Args:
     - input_units: The last dimension of the input data
     - ENCODED_DIM: The desired lower dimension embedding you want to generate
                    Or to say, the appropriate latent representation of teh original input
     - sparsity: The user defined value of the desirable percentage of the activated portion across the training sample for each node.
     - regu: The strength of the sparsity regularization
    """
    
    def __init__(self,input_units, ENCODED_DIM, sparsity, regu, **kwargs):
        super(AE,self).__init__(**kwargs)
        self.input_units=input_units
        self.units = ENCODED_DIM
        self.sparsity = sparsity
        self.strength = regu
        

        self.layer1 = Encoder(self.units,self.sparsity,self.strength)
        self.layer2 = Decoder(self.input_units)
        
    def call(self,inputs):
        
        h=self.layer1(inputs)
        return self.layer2(h)

    
# After customizing the basic Model and autoencoder
# Establish the training and inference method which resembles the LeNet_train.py   
        
class Autoencoder:
    
    """
    >> Key function <<
    
     - Instance construction:
       Args:
        - ENCODED_DIM: The desired lower dimension embedding you want to generate
        - sparsity: The user defined value of the desirable percentage of the activated portion across the training sample for each node.
        - regu: The strength of the sparsity regularization
        - learnng_rate
        - batch_size: default 64
        - epochs: default 30
        - patience: (default 10) how many epochs we should wait for to early stop the training process 
                                 after the validation loss doesn't decrease anymore. 
        - patience_lr: (default 5) how many epochs we should wait for to shrink the learning rate by half
                                   after the validation loss doesn't decrease anymore.
          
     - 2 method call of Autoencoder class:
     (First, let the constructed instance to call "fit" method to the training data
      Next, infer the transformed representation using "transform" method to fit training or testing data)
     
     1. fit: 
         Args: inputs -- Training sample data of the shape (Number of sample, dimension of input feature space)
         
     2. Transform:
         Args: inputs -- Initial training sample or testing sample
               output -- Transformed training sample or testing sample (Number of sample, Encoded dimension)
    """
    
    def __init__(self, ENCODED_DIM, sparsity, regu,
                learning_rate, batch_size = 64 , epochs=30, patience=10, patience_lr=5):
        
        self.hidden_units=ENCODED_DIM
        self.sparsity = sparsity
        self.strength = regu
        self.lr = learning_rate
        self.bs = batch_size
        self.epochs = epochs
        self.patience = patience
        self.patience_lr = patience_lr
    
    def fit(self,inputs):
        
        # Get initial input dimension
        num=inputs.shape[0]
        input_shape=inputs.shape[-1] 
        self.input_units = input_shape
        
        # Construct an AE class model instance
        single_model = AE(input_shape,self.hidden_units, self.sparsity,self.strength)
        
        # Define the loss and metrics used to train the model              
        mse = tf.keras.losses.MeanSquaredError()
        loss_metric = tf.keras.metrics.Mean()
        
        
     # Training the model
        # Split the training and validation set
        x_train, x_train, x_vali, x_vali = ms.train_test_split(inputs,
                                                               inputs,
                                                               test_size = 0.2,
                                                               random_state = 42)
        
        train_dataset = tf.data.Dataset.from_tensor_slices(x_train)
        train_dataset = train_dataset.shuffle(buffer_size=1024).batch(self.bs)

        epochs = self.epochs
        
        # Initialize the variables used as the standard to early stop the training or to reduce the learning rate
        x_vali_min = np.inf
        count = 0 # Define it to use validation set to set early_stopping critieron
        count_lr = 0 # use validation set to shrink the learning rate by half with certain patience level
        
        # Iterate over epochs.
        for epoch in range(epochs):
            print("Start of epoch %d" % (epoch,))
            epoch_loss = 0
            opt = tf.keras.optimizers.RMSprop(learning_rate=self.lr)
            # RMSprop is best. Adam is good sometimes. Others have difficulty converging
            
            for step, x_batch_train in enumerate(train_dataset):
                
                with tf.GradientTape() as tape:
                    reconstructed = single_model(x_batch_train)
                    # Compute reconstruction loss
                    loss = mse (x_batch_train, reconstructed)
                    loss += sum(single_model.losses)  # Add KLD regularization loss
                
                grads = tape.gradient(loss, single_model.trainable_weights)
                opt.apply_gradients(zip(grads, single_model.trainable_weights))

                #loss_metric(loss)
            
            # Use the generated validation loss to compare with the stored minimum validation loss by far.
            vali_loss = mse(single_model.predict(x_vali),x_vali) 
            
            # Print out the training loss and validation loss of each epochs
            print("train_loss = %.6f" % (loss_metric.result()), end=" ")
            print("val_loss = %.6f" % (vali_loss))
            
            
            # Inside each epochs, after training an epoch, need to compute the updated patience level
            if count_lr >= self.patience_lr:
                self.lr=0.5*self.lr  # Shrink the learning rate by half 
             
            # Break the for loop of training if necessary
            if count >= self.patience:
                print("The validation loss doesn't decrease within {:d} epochs: Early stop the training".format(self.patience))
                break
            
            if vali_loss < x_vali_min:
                x_vali_min = vali_loss
                
                # save weights of the model of the optimal validation loss up to now 
                self.save_weights = single_model.layer1.get_weights()
                count = 0
                count_lr=0
                
            else:
                count += 1
                count_lr+=1
    
    # Inference method
    def transform(self,inputs):
        
        # Refer from the saved weights of the encoded layer
        # Recompute the inputs by walking through the trained encoder layer to get the inferred embeded representation
        W1,b1,W2,b2 = self.save_weights[:4]
        h = tf.keras.layers.LeakyReLU(alpha=0.3)(inputs@W1+b1)
        optimal_layer_outputs = tf.keras.layers.LeakyReLU(alpha=0.3)(h@W2+b2)
        
        return optimal_layer_outputs
    