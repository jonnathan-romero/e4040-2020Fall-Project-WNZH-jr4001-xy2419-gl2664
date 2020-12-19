import tensorflow as tf
from tensorflow.keras.layers import LSTM, Dense, Concatenate, Input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint,ReduceLROnPlateau


class MLP:
    
    def __init__(self, dim, type='classification'):
        '''
        MLP model
        type: 'classification' or 'regression'. For classification problem, y is labelde as one of the 5 quintiles of stock return,
              and categorical cross-entropy lossis used for back-propagation. For regression problem, y is the rank score of stock return
              (from 0 to 1), and (pointwise) binary cross-entropy loss is applied instead.
        dim: input dimension
        '''
        
        self.type=type
        cross_sectional_inputs = Input(shape=(dim,))
        h1=Dense(128, activation='relu')(cross_sectional_inputs)
        h2=Dense(64, activation='relu')(h1)
        h3=Dense(32, activation='relu')(h2)
        outputs=Dense(5, activation='softmax')(h3) if type=='classification' else Dense(1, activation='sigmoid')(h3)

        self.model=tf.keras.Model(inputs=cross_sectional_inputs, outputs=outputs)
     
    
    def fit(self, x, y, save_path, opt=None, earlystopping=None, lr_reduce=None):
        '''
        fit MLP model given data, and save the best model
        '''
        if not opt:
            opt=tf.keras.optimizers.Adam(learning_rate=0.005) # set optimizer
        if not earlystopping:
            earlystopping=EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='min') # early stopping
        if not lr_reduce:
            lr_reduce=ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, verbose=1, min_delta=5e-5, mode='min') # set automatic lr
            
        loss='CategoricalCrossentropy' if self.type=='classification' else 'binary_crossentropy'
        metrics='accuracy' if self.type=='classification' \
            else ['binary_crossentropy','mean_squared_error','mean_absolute_error','mean_absolute_percentage_error']
        self.model.compile(optimizer=opt, loss=loss ,metrics=metrics)
        mcp_save=ModelCheckpoint(save_path, save_best_only=True, monitor='val_loss', mode='min')                                                        
        history=self.model.fit(x, y, batch_size=256, epochs=100, verbose=1, 
                              callbacks=[earlystopping, mcp_save, lr_reduce],
                              validation_split=0.1)                   
        
        return history
    
    def predict(self, xtest):
        return self.model.predict(xtest)
                  
class Hybrid:
    
    def __init__(self, dim, type='classification'):
        '''
        MLP+LSTM model
        type: 'classification' or 'regression'. For classification problem, y is labelde as one of the 5 quintiles of stock return,
              and categorical cross-entropy lossis used for back-propagation. For regression problem, y is the rank score of stock return
              (from 0 to 1), and (pointwise) binary cross-entropy loss is applied instead.
        dim: input dimension (list of 2: cross_sectional_dim + time_series_dim)
        '''
        
        self.type=type
        cross_sectional_inputs = Input(shape=(dim[0],)) 
        time_series_inputs = Input(shape=(dim[1],1))
        

        h1_rets=LSTM(units=50, return_sequences=True)(time_series_inputs)
        time_series_output=LSTM(units=30, return_sequences=False)(h1_rets) 

        combined_features = Concatenate()([cross_sectional_inputs, time_series_output]) # concatenate LSTM and MLP layer

        h1=Dense(128, activation='relu')(combined_features)
        h2=Dense(64, activation='relu')(h1)
        h3=Dense(32, activation='relu')(h2)
        outputs=Dense(5, activation='softmax')(h3) if type=='classification' else Dense(1, activation='sigmoid')(h3)

        self.model=tf.keras.Model(inputs=[cross_sectional_inputs,time_series_inputs], outputs=outputs)
   
        
    def fit(self, x, y, save_path, opt=None, earlystopping=None, lr_reduce=None):
        '''
        fit  MLP+LSTM model given data, and save the best model
        '''
        
        if not opt:
            opt=tf.keras.optimizers.Adam(learning_rate=0.005)
        if not earlystopping:
            earlystopping=EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='min')
        if not lr_reduce:
            lr_reduce=ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, verbose=1, min_delta=5e-5, mode='min')   
            
        loss='CategoricalCrossentropy' if self.type=='classification' else 'binary_crossentropy'
        metrics='accuracy' if self.type=='classification' \
            else ['binary_crossentropy','mean_squared_error','mean_absolute_error','mean_absolute_percentage_error']
        self.model.compile(optimizer=opt, loss=loss ,metrics=metrics)
        mcp_save=ModelCheckpoint(save_path, save_best_only=True, monitor='val_loss', mode='min')                                                        
        history=self.model.fit(x, y, batch_size=256, epochs=100, verbose=1, 
                              callbacks=[earlystopping, mcp_save, lr_reduce],
                              validation_split=0.1)                   
        
        return history   
    
    def predict(self, xtest):
        return self.model.predict(xtest)
                                                                