from sklearn.model_selection import train_test_split
from sklearn.externals import joblib
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers.embeddings import Embedding
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import Conv2D
from keras.layers import Conv1D
from keras.layers import MaxPooling1D
from keras.layers import MaxPooling2D
from keras.layers import GlobalMaxPooling1D
from keras.layers import GlobalMaxPooling2D
from keras.layers import AveragePooling2D
from keras.layers import AveragePooling1D
from keras.optimizers import Adam
from keras.optimizers import SGD
from keras.optimizers import Adagrad
from keras.layers import Reshape
from keras.layers import Input
import numpy as np
from keras.preprocessing.text import one_hot
from random import choice
from random import *
from keras.layers.normalization import BatchNormalization
from keras.layers import LeakyReLU
import tensorflow as tf
import keras.backend as K

class Generator:
    
    
    def __init__(self,batch_size):
        self.ids_train = []
        self.ids_validate = []
        self.batch_size = batch_size
        self.create_list()
        
    def create_list(self):
        for i in range(int(self.batch_size*0.6)):
            self.ids_train.append(choice([i for i in range(0,self.batch_size) if i not in self.ids_train]))
        for i in range(self.batch_size):
            if i not in self.ids_train:
                self.ids_validate.append(i)
            
    def generate(self,phase):
      
        while 1:
            if phase =='train':
                for i in self.ids_train:
                    X = np.load('images/images'+str(i)+'.npy')
                    y = np.load('output/output'+str(i)+'.npy')
                    
                    yield X,y
                    
            if phase =='validate':
                for i in self.ids_validate:
                    X = np.load('images/images'+str(i)+'.npy')
                    y = np.load('output/output'+str(i)+'.npy')
                    
                    yield X,y
   

def loss_function(y_true,y_pred):
    object_true = y_true[:,:,:,0]
    object_pred = y_pred[:,:,:,0]
    
    width_true = y_true[:,:,:,1]
    width_pred = y_pred[:,:,:,1]
    
    height_true = y_true[:,:,:,2]
    height_pred =y_pred[:,:,:,2]
    
    X_true = y_true[:,:,:,3]
    X_pred =y_pred[:,:,:,3]
    
    Y_true = y_true[:,:,:,4]
    Y_pred = y_pred[:,:,:,4]
    
    loss1 = np.abs(tf.scalar_mul(5.0,tf.reduce_sum(tf.multiply(object_true,(tf.squared_difference(X_pred,X_true) + tf.squared_difference(Y_pred,Y_true))))))
    loss2 = np.abs(tf.scalar_mul(5.0,tf.reduce_sum(tf.multiply(object_true,(tf.squared_difference(tf.sqrt(width_pred),tf.sqrt(width_true)) + tf.squared_difference(tf.sqrt(height_pred),tf.sqrt(height_true)))))))
    loss3 = np.abs(tf.reduce_sum(tf.multiply(object_true,(tf.squared_difference(object_pred,object_true)))))
    loss4 = np.abs(tf.scalar_mul(0.5,(tf.reduce_sum(tf.multiply((object_true*-1+1),(tf.squared_difference(object_pred,object_true)))))))
    
    return loss1+loss2+loss3+loss4


def get_coord(X,Y,H,W):
    x2 = X + W
    y2 = Y + H
    
    return x2,y2

def loss_function2(y_true,y_pred):
    object_true = y_true[:,:,:,0]
    object_pred = y_pred[:,:,:,0]
    
    width_true = y_true[:,:,:,1]
    width_pred = y_pred[:,:,:,1]
    
    height_true = y_true[:,:,:,2]
    height_pred =y_pred[:,:,:,2]
    
    X_true = y_true[:,:,:,3]
    X_pred =y_pred[:,:,:,3]
    
    Y_true = y_true[:,:,:,4]
    Y_pred = y_pred[:,:,:,4]
    
    x2_true,y2_true = get_coord(X_true,Y_true,height_true,width_true)
    x2_pred,y2_pred = get_coord(X_pred,Y_pred,height_pred,width_pred)
    
    xi1 = tf.maximum(X_true,X_pred)
    yi1 = tf.maximum(Y_true,X_true)
    xi2 = tf.minimum(x2_true,x2_pred)
    yi2 = tf.minimum(y2_true,y2_pred)

    
    zeros = tf.zeros([32,7,7])
    
    inter_area = tf.maximum(zeros,xi2-xi1)*tf.maximum(zeros,yi2-yi1)
    box_area1 = (x2_true-X_true)*(y2_true-Y_true)
    box_area2 = (x2_pred-X_pred)*(y2_pred-Y_pred)
    
    union_area = box_area1 + box_area2 - inter_area
    iou = inter_area/union_area
    nb_obj = tf.reduce_sum(object_true)
    nb_noobj = tf.reduce_sum(object_true*-1+1)
    
    loss1 = 5*tf.reduce_sum(object_true*((X_pred-X_true)**2+(Y_pred-Y_true)**2)) / (nb_obj +1e-6)/2.
    loss2 = 5*tf.reduce_sum(object_true*((tf.sqrt(width_pred)-tf.sqrt(width_true))**2 + (tf.sqrt(height_pred)-tf.sqrt(height_true))**2))/ (nb_obj +1e-6)/2.
    loss3 = 5*tf.reduce_sum(object_true*(object_pred-iou)**2) / (nb_obj +1e-6)/2.
    loss4 = 0.5*tf.reduce_sum(((object_true*-1)+1)*(object_pred-object_true)**2) / (nb_noobj +1e-6)/2.
    
#    confidence = tf.multiply(iou,object_pred)
#    losslog1 = -1*tf.log(tf.multiply(object_true,confidence)+1e-9)
#    losslog2 = -1*tf.log(1-(tf.multiply((object_true*-1+1),object_pred)))
    
    
    return loss1+loss2+loss3+loss4
             
def SSQE(y_true,y_pred):
    return K.sum(K.square(y_pred - y_true), axis=-1)

batch_size = 5
                
generator = Generator(batch_size)
training_generator = generator.generate('train')
validation_generator = generator.generate('validate')

model = Sequential()


model.add(Conv2D(64,(7,7),padding="SAME",input_shape=(448,448,3),activation="relu",strides=2))
model.add(MaxPooling2D((2,2),strides=2,padding='SAME'))

model.add(Conv2D(64,(3,3),padding="SAME",activation="relu"))
model.add(MaxPooling2D((2,2),strides=2,padding='SAME'))

model.add(Conv2D(64,(3,3),padding="SAME",activation="relu"))
model.add(MaxPooling2D((2,2),strides=2,padding='SAME'))

model.add(Conv2D(64,(3,3),padding="SAME",activation="relu"))
model.add(MaxPooling2D((2,2),strides=2,padding='SAME'))

model.add(Conv2D(128,(3,3),padding="SAME",activation="relu"))
model.add(MaxPooling2D((2,2),strides=2,padding='SAME'))

#model.add(Conv2D(32,(7,7),padding="SAME",input_shape=(608,608,3)))
##model.add(BatchNormalization())
#model.add(LeakyReLU(alpha=0.1))
#model.add(MaxPooling2D(2,2))
#
#model.add(Conv2D(32,(3,3),padding="SAME"))
##model.add(BatchNormalization())
#model.add(LeakyReLU(alpha=0.1))
#model.add(MaxPooling2D(2,2))
#
#model.add(Conv2D(32,(1,1),padding="SAME"))
##model.add(BatchNormalization())
#model.add(LeakyReLU(alpha=0.1))
#model.add(Conv2D(64,(3,3),padding="SAME"))
##model.add(BatchNormalization())
#model.add(LeakyReLU(alpha=0.1))
#model.add(Conv2D(64,(1,1),padding="SAME"))
##model.add(BatchNormalization())
#model.add(LeakyReLU(alpha=0.1))
#model.add(Conv2D(64,(3,3),padding="SAME"))
##model.add(BatchNormalization())
#model.add(LeakyReLU(alpha=0.1))
#model.add(MaxPooling2D(2,2))
#
#model.add(Conv2D(32,(1,1),padding="SAME"))
##model.add(BatchNormalization())
#model.add(LeakyReLU(alpha=0.1))
#model.add(Conv2D(64,(3,3),padding="SAME"))
##model.add(BatchNormalization())
#model.add(LeakyReLU(alpha=0.1))
#model.add(Conv2D(32,(1,1),padding="SAME"))
##model.add(BatchNormalization())
#model.add(LeakyReLU(alpha=0.1))
#model.add(Conv2D(64,(3,3),padding="SAME"))
##model.add(BatchNormalization())
#model.add(LeakyReLU(alpha=0.1))
#model.add(Conv2D(32,(1,1),padding="SAME"))
##model.add(BatchNormalization())
#model.add(LeakyReLU(alpha=0.1))
#model.add(Conv2D(64,(3,3),padding="SAME"))
##model.add(BatchNormalization())
#model.add(LeakyReLU(alpha=0.1))
#model.add(Conv2D(32,(1,1),padding="SAME"))
##model.add(BatchNormalization())
#model.add(LeakyReLU(alpha=0.1))
#model.add(Conv2D(64,(3,3),padding="SAME"))
##model.add(BatchNormalization())
#model.add(LeakyReLU(alpha=0.1))
#model.add(Conv2D(64,(1,1),padding="SAME"))
##model.add(BatchNormalization())
#model.add(LeakyReLU(alpha=0.1))
#model.add(Conv2D(128,(3,3),padding="SAME"))
##model.add(BatchNormalization())
#model.add(LeakyReLU(alpha=0.1))
#model.add(MaxPooling2D(2,2))
#
#model.add(Conv2D(64,(1,1),padding="SAME"))
##model.add(BatchNormalization())
#model.add(LeakyReLU(alpha=0.1))
#model.add(Conv2D(128,(3,3),padding="SAME"))
##model.add(BatchNormalization())
#model.add(LeakyReLU(alpha=0.1))
#model.add(Conv2D(64,(1,1),padding="SAME"))
##model.add(BatchNormalization())
#model.add(LeakyReLU(alpha=0.1))
#model.add(Conv2D(128,(3,3),padding="SAME"))
##model.add(BatchNormalization())
#model.add(LeakyReLU(alpha=0.1))
#model.add(Conv2D(64,(3,3),padding="SAME"))
##model.add(BatchNormalization())
#model.add(LeakyReLU(alpha=0.1))
#model.add(MaxPooling2D(2,2))
#
model.add(Conv2D(128,(3,3),padding="SAME",activation="relu"))
#model.add(BatchNormalization())
#model.add(LeakyReLU(alpha=0.1))
model.add(Conv2D(128,(3,3),padding="SAME",activation="relu"))
#model.add(BatchNormalization())
#model.add(LeakyReLU(alpha=0.1))
model.add(Conv2D(1024,(1,1),activation='relu',padding="SAME"))
##model.add(Dropout(0.5))
model.add(Conv2D(5,(1,1),activation='softmax',padding="SAME"))


# compile the model
optimizer = Adam(lr=0.0001)
model.compile(optimizer=optimizer, loss=loss_function2, metrics=['acc'])
# summarize the model
print(model.summary())


# fit the model
model.fit_generator(generator = training_generator,
                    steps_per_epoch = batch_size*0.7,
                    validation_data = validation_generator,
                    validation_steps = batch_size*0.3,
                    max_queue_size=1,
                    epochs=50)

# evaluate the model
#loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
#print('Accuracy: %f' % (accuracy*100))