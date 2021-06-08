import tensorflow as tf
from encoder_model import Encoder

class LoanGrader(tf.keras.Model):

    def __init__(self, num_encoder_layer, d_model, num_heads, dff, max_positional_encoding=34, input_dims=85267,  rate=0.1, **kwargs):
        super(LoanGrader, self).__init__(**kwargs)

        self.encoder= Encoder(num_encoder_layer, d_model, num_heads, dff, input_dims, max_positional_encoding, rate=rate )


        self.flatten= tf.keras.layers.Flatten()
        self.cancat= tf.keras.layers.Concatenate(axis=-1)

        
        self.dense1= tf.keras.layers.Dense(128, activation='relu')
        self.dense2= tf.keras.layers.Dense(64, activation= 'relu')
        self.dense3= tf.keras.layers.Dense(64, activation='relu')
        self.dense4=tf.keras.layers.Dense(32, activation='relu')
        self.dense5= tf.keras.layers.Dense(7, activation='softmax')

        self.class_weights=None

       
        self.optimizer= None
        
        self.variable_grouped=False

        self.epochs=0

        self.training_history= {
                                'training_loss': [],
                                'training_accuracy': [],
                                'validation_loss': [],
                                'validation_accuracy' : []
        }

    

    def call(self, X, sequence, training=False):
        """
        Performs forward pass
        """
        mask= create_padding_mask(sequence)
        x= self.encoder(sequence, mask=mask)
        x= self.flatten(x)

        X=self.batch_normalization(X, training=training)
        x= self.cancat([x, X])

        x= self.dense1(x)
        x= self.dense2(x)
        x= self.dense3(x)
        x= self.dense4(x)
        x= self.dense5(x)        

        return x

    
    def adjust_labels(self, labels, predictions):
   
        labels = tf.cast(labels, tf.int32)
        if len(predictions.shape) == len(labels.shape):
            labels = tf.squeeze(labels, [-1])
        return labels, predictions

    
    def validate_rank(self, labels, predictions, weights):
        if weights is not None and len(weights.shape) != len(labels.shape):
            raise RuntimeError(
                ("Weight and label tensors were not of the same rank. weights.shape "
                "was %s, and labels.shape was %s.") %
                (predictions.shape, labels.shape))
        if (len(predictions.shape) - 1) != len(labels.shape):
            raise RuntimeError(
                ("Weighted sparse categorical crossentropy expects `labels` to have a "
                "rank of one less than `predictions`. labels.shape was %s, and "
                "predictions.shape was %s.") % (labels.shape, predictions.shape))


    def loss(self, labels, predictions, class_weights=None, from_logits=False):
    
        # When using these functions with the Keras core API, we will need to squeeze
        # the labels tensor - Keras adds a spurious inner dimension.
        labels= tf.cast(labels, tf.int32)
        weights = tf.gather(class_weights, labels)
        

        labels, predictions = self.adjust_labels(labels, predictions)
        self.validate_rank(labels, predictions, weights)

        example_losses = tf.keras.losses.sparse_categorical_crossentropy(
            labels, predictions, from_logits=from_logits)

        if weights is None:
            return tf.reduce_mean(example_losses)
        weights = tf.cast(weights, predictions.dtype)
        return tf.math.divide_no_nan(
            tf.reduce_sum(example_losses * weights), tf.reduce_sum(weights))

    def compile(self,  optimizer, encoder_optimizer=None):
        self.optimizer= optimizer
        
        if encoder_optimizer != None:
            self.encoder_optimizer= encoder_optimizer
            
    @tf.function
    def grad(self, x_train, train_sequence, y_train):
        
        with tf.GradientTape() as tape:
            y_pred= self.call(x_train, train_sequence, training=True)
            loss_val= self.loss(y_train, y_pred, class_weights=self.class_weights)
            gradients= tape.gradient(loss_val, self.trainable_variables )
            

            self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
            
            return loss_val, y_pred

    @tf.function
    def evaluate(self, test_data, return_loss=False):
        test_loss= tf.keras.metrics.Mean()
        test_accuraccy=tf.keras.metrics.SparseCategoricalAccuracy()
        for x, sequence, y in test_data:

            y_pred= self.call(x, sequence, training=False)
            
            

            test_accuracy.update_state(y, y_pred)
            test_loss.update_state(self.loss(y, y_pred, class_weights=self.class_weights))

        if retrun_loss :
            return test_accuracy.results(), test_loss.results()
        else:
            return test_accuracy.results()
            
         

    
    def train(self, training_data, epochs, validation_data, class_weights=[1,1,1,1,1,1,1]):
        
        
        self.class_weights=class_weights

        steps_per_epoch= training_data.cardinality().numpy()
        steps_per_epoch_val= validation_data.cardinality().numpy()
        
        for epoch in range(epochs):
            
            start=time.time()
            print('Epoch ', epoch+1, ' :')
            this_epoch=display('Starting Epoch %d'%(epoch+1), display_id=True)
            train_loss = tf.keras.metrics.Mean(name='train_loss')
            train_accuracy =tf.keras.metrics.SparseCategoricalAccuracy()

            step=0
            for x,sequence, y in training_data:
                
                loss_val, y_pred =  self.grad(x, sequence, y)

                train_loss.update_state(loss_val)
                train_accuracy.update_state(y, y_pred)

                step+=1

                if step % 10== 0:
                    this_epoch.update('     {}% Compleated  | Loss {:.4f} | Accuracy {:.2f}% | Time: {} mins'.format( int((step/steps_per_epoch)*100),
                    train_loss.result(), train_accuracy.result()*100, (time.time()-start)//60))

            else:
                this_epoch.update('     {}% Compleated  | Loss {:.4f} | Accuracy {:.2f}% | Time: {} mins'.format( int((step/steps_per_epoch)*100),
                train_loss.result(), train_accuracy.result()*100, (time.time()-start)//60))
                self.epochs +=1


            self.training_history['training_loss'].append(train_loss.result())
            self.training_history['training_loss'].append(train_accuracy.result())
            
            val_start=time.time()
            step=0

            val_loss=tf.keras.metrics.Mean()
            val_accuracy=tf.keras.metrics.SparseCategoricalAccuracy()


            this_epoch_val= display('Evaluating on validation set', display_id=True)

            for x,sequence, y in validation_data:
            
                y_pred= self.call(x, sequence, training=False)

                loss_val= self.loss(y, y_pred, class_weights=self.class_weights)
                

                val_loss(loss_val)
                val_accuracy(y, y_pred)

                step+=1

                if step % 10==0:
                    this_epoch_val.update('   Validation: {}% Compleated | Loss {:.4f} | Accuracy {:.2f}% | Time: {} mins'.format(int((step/steps_per_epoch_val)*100),
                    val_loss.result(), val_accuracy.result()*100, (time.time()- val_start)//60))
            
            else:
                
                this_epoch_val.update('   Validation: {}% Compleated | Loss {:.4f} Accuracy | {:.2f}% | Time: {} mins'.format(int((step/steps_per_epoch_val)*100),
                val_loss.result(), val_accuracy.result()*100, (time.time()- val_start)//60))

                

            self.training_history['validation_loss'].append(val_loss.result())
            self.training_history['validation_accuracy'].append(val_accuracy.result())




