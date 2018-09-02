#Let us make a simple server ,which let us interact with our model from a browser

import flask
import pickle
import numpy as np
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
import models.multichannel_drop as md
import keras.backend as K
app=flask.Flask(__name__,static_url_path="/static")
import matplotlib.pyplot as plt
from scipy.misc import imresize

vocab_size=15000
max_seq_len=250
word2index=pickle.load(open("../weights/word2index.p","rb"))
embedding_matrix=np.load("../weights/embedding_matrix.npy")

count=0
size=0
#*********************************************************************************************#
print("loading model .....")
model=md.multichannel(nb_classes=1,vocab_size=vocab_size,input_len=max_seq_len,matrix=embedding_matrix,drop=False)
model.load_weights("../weights/polarity.h5")
cache=model.layers[-3].output
sess=K.get_session()
#based on this argument we are visualising 
#see visualisation.ipynb for more details

arg=np.argmax(model.layers[-1].get_weights()[0][:100])

print("model loaded successfully....")
#print(model.summary())

def preprocess(text):
    global size
    #preprocess  query to make it model compatible
    temp=[]
    for word in text.strip().split():
        
        if word in word2index: 
            if word2index[word] < vocab_size:
                temp.append(word2index[word])
            else:
                temp.append(vocab_size+1)
        
    size=len(temp)
    return pad_sequences(np.array([temp]).reshape(1,-1),max_seq_len)


@app.route("/predict",methods=["POST"])
def predict():
    global model
    global size
    global count
    dic={"success":False}
    if flask.request.method=="POST":
        
        if flask.request.form.get("query"):
            query=flask.request.form["query"]
            
            
            data=preprocess(query)
            
            assert data.shape==(1,max_seq_len)

            pred,activations=sess.run([model.output,cache],feed_dict={model.input:data})
            pred=pred[0][0]
            activation=activations[0,:,arg] #of shape(250,)
            name="static/img_"+str(count)+".png"
            plt.imsave(name,imresize(activation[-size:].reshape(1,-1),(200,100*size)))
            
            dic["positive"]=str(pred)
            dic["negative"]=str(1-pred)
            dic["id"]=str(count)
            dic["success"]=True
           
            count+=1
    return flask.jsonify(dic)


if __name__=="__main__":
    app.run("127.0.0.1","8000")
