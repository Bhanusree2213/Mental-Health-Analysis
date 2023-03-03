import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
from gevent.pywsgi import WSGIServer
# Create flask apps
flask_app = Flask(__name__,template_folder="templates")
model = pickle.load(open("model.pkl", "rb"))



@flask_app.route("/")
def Home():
    return render_template("index.html")

@flask_app.route("/predict", methods = ["POST"])
def predict():
    features = [x for x in request.form.values()]
    features = np.array(features)
    no_of_factors=len(features)-10
    factors=[0]*10
    index={"Friends":0,"Education":1,"Family":2,"Strangers":3,"Health":4,"Relatives":5,"Career":6,"Lover":7,"None":8,"Marriage":9}
    for i in features[11:]:
        factors[index[i]]=1 
    inp=[]
    inp.append(int(features[0]))#agegroup
    inp.append(int(features[1]))#gender
    inp.append(int(features[3]))#physical
    inp.append(int(features[4]))#sleep
    inp.append(int(features[6]))#bad
    inp.append(int(features[7]))#relax
    inp.append(int(features[9]))#anxiet
    inp.append(int(features[8]))#suicidal
    inp.append(int(features[10]))#fam
    inp.extend(factors)
    inp.append(0)
    ans=model.predict([inp])
    res=""
    if ans==1:
        res+="You seem to Mentally Depressed"
    else:
        res+="You are not Depressed"
    return render_template("index.html", Prediction = "{}".format(res))
    
if __name__ == "__main__":
    flask_app.run(debug=True) 