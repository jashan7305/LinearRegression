from flask import Flask, render_template, redirect, url_for, request
from prediction import predict_with_toefl, predict_without_toefl

app = Flask(__name__)

@app.route("/", methods=["POST", "GET"])
def home():
    if request.method == "POST":
        print("Request method is POST")
        cgpa = request.form.get("cgpa")
        gre = request.form.get("gre")
        toefl = request.form.get("toefl")
        lor = request.form.get("lor")
        uni_rating = request.form.get("uniRating")
        research = request.form.get("research")

        if toefl == 0:
            prediction = predict_without_toefl(gre, lor, cgpa, uni_rating, research)
            print("Prediction without TOEFL:", prediction)
        else:
            prediction = predict_with_toefl(gre, toefl, lor, cgpa, uni_rating, research)
            print("Prediction with TOEFL:", prediction)

        return render_template("index.html", prediction=prediction)
    else:
        return render_template("index.html", prediction=None)

@app.route("/predict", methods=["post", "get"])
def predict():
    pass


if __name__ == "__main__":
    app.run(debug=True)