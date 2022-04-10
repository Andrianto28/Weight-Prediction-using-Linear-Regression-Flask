from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)


@app.route("/", methods=['GET', 'POST'])
def weight_prediction():
    if request.method == 'GET':
        return render_template("weight-prediction.html")
    elif request.method == 'POST':
        print(dict(request.form))
        weight_features = dict(request.form).values()
        A= []
        for x in weight_features:
            if x == 'Male':
                x = 1.0
            elif x == 'Female':
                x = -1.0
            else:
                x = float(x)
            A.append(x)
        weight_features = np.array(A)
        model, std_scaler = joblib.load("model-development/weight-classification-using-linear-regression.pkl")
        weight_features = std_scaler.transform([weight_features])
        print(weight_features)
        result = model.predict(weight_features)
        return render_template('weight-prediction.html', result=np.round(result[0],2))
    else:
        return "Unsupported Request Method"


if __name__ == '__main__':
    app.run(port=5000, debug=True)