#multichannel architecture based on the research paper "Convolutional Neural Networks for Sentence Classification" (http://arxiv.org/abs/1408.5882)

import keras 
from keras.layers import Dense,GlobalAveragePooling1D,Conv1D,Embedding,add,concatenate
from keras.layers import Input
from keras.models  import Model
import numpy as np
from keras.layers import Dropout
from keras.layers import SpatialDropout1D


def multichannel(nb_classes,vocab_size,input_len,embedding_dim=100,matrix=None,activation="sigmoid",drop=False):

    inp=Input(shape=(input_len,))

    if matrix is None:
        matrix=np.random.rand(vocab_size+1,embedding_dim)

    assert matrix.shape == (vocab_size+1,embedding_dim)


    embed_layer_1=Embedding(vocab_size+1,embedding_dim,input_length=input_len)
    embed_layer_1.build(input_shape=(input_len,))
    embed_layer_1.set_weights([matrix])

    embed_layer_2=Embedding(vocab_size+1,embedding_dim,input_length=input_len)
    embed_layer_2.build(input_shape=(input_len,))
    embed_layer_2.set_weights([matrix])
    embed_layer_2.trainable=False


    window_sizes=[2,3,5]
    dic={}
    for size in window_sizes:
        #addding all the layer instances to a dic for easy handling 
        dic[size]=Conv1D(kernel_size=size,strides=1,padding="same",filters=100,activation="relu")


    dropout_layer=SpatialDropout1D(0.2)
    
    x1=embed_layer_1(inp)
    x2=embed_layer_2(inp)

    #notice we are sharing the layers , for both x1,x2 we are using the same weights
    y1_2=dic[2](x1)
    y1_2=dropout_layer(y1_2)
    y1_3=dic[3](x1)
    y1_3=dropout_layer(y1_3)
    y1_5=dic[5](x1)
    y1_5=dropout_layer(y1_5)

    y2_2=dic[2](x2)
    y2_2=dropout_layer(y2_2)
    y2_3=dic[3](x2)
    y2_3=dropout_layer(y2_3)
    y2_5=dic[5](x2)
    y2_5=dropout_layer(y2_5)


    concat_1=concatenate([y1_2,y1_3,y1_5])
    concat_2=concatenate([y2_2,y2_3,y2_5])

    add_1=add([concat_1,concat_2])
    pool_1=GlobalAveragePooling1D()(add_1)
    if drop==True:
        #add dropout
        pool_1=Dropout(0.5)(pool_1)


    final_1=Dense(nb_classes,activation=activation)(pool_1)

    model=Model([inp],[final_1])

    return model


