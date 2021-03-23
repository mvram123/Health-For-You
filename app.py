from flask import Flask, render_template, request, sessions, redirect, url_for
import pickle
import numpy as np
import random
from flask_sqlalchemy import SQLAlchemy

app = Flask(__name__)

app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///data.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)


class User(db.Model):

    id = db.Column(db.Integer, primary_key=True )
    age = db.Column(db.Float)
    bp = db.Column(db.Float)
    sg = db.Column(db.Float)
    al = db.Column(db.Float)
    su = db.Column(db.Float)
    rbc = db.Column(db.Float)
    pc = db.Column(db.Float)
    pcc = db.Column(db.Float)
    ba = db.Column(db.Float)
    bu = db.Column(db.Float)
    sc = db.Column(db.Float)
    sod = db.Column(db.Float)

    def __repr__(self):
        return '<User %r>' % self.age


@app.route('/')
def home():
    return render_template('index.html')

# ---------------Cancer Prediction-------------------------


@app.route('/cancer')
def cancer():
    return render_template('work.html')


@app.route('/cancer_predict', methods=['POST'])
def cancer_predict():

    model = pickle.load(open('Models/cancer.pkl', 'rb'))
    data_array = list()

    radius_mean = float(request.form['radius_mean'])
    texture_mean = float(request.form['texture_mean'])
    smoothness_mean = float(request.form['smoothness_mean'])
    compactness_mean = float(request.form['compactness_mean'])
    symentry_mean = float(request.form['symentry_mean'])
    fractal_dim_mean = float(request.form['fractal_dim_mean'])
    texture_se = float(request.form['texture_se'])
    smoothness_se = float(request.form['smoothness_se'])
    symentric_se = float(request.form['symentric_se'])
    symentric_worst = float(request.form['symentric_worst'])

    data_array = data_array + [radius_mean, texture_mean, smoothness_mean,
                               compactness_mean, symentry_mean, fractal_dim_mean,
                               texture_se, smoothness_se, symentric_se, symentric_worst]

    data = np.array([data_array])
    value = int(model.predict(data))

    if value == 0:
        a = 'Benign'
    else:
        a = 'Malignant'

    return render_template('cancer_predict.html', value=a)

# ---------------Diabetes Prediction-------------------------


@app.route('/diabetes')
def diabetes():
    return render_template('diabetes.html')


@app.route('/diabetes_predict', methods=['POST'])
def diabetes_predict():

    model = pickle.load(open('Models/diabetes.pkl', 'rb'))
    data_array = list()

    pregnancies = float(request.form['pregnancies'])
    glucose = float(request.form['glucose'])
    bp = float(request.form['bp'])
    skin_thickness = float(request.form['skin_thickness'])
    insulin = float(request.form['insulin'])
    bmi = float(request.form['bmi'])
    dpf = float(request.form['dpf'])
    age = float(request.form['age'])

    data_array = data_array + [pregnancies, glucose, bp, skin_thickness,
                               insulin, bmi, dpf, age]

    data = np.array([data_array])
    value = int(model.predict(data))

    if value == 0:
        a = "Congratulations!! You Don't Have Diabetes"
    else:
        a = "Sorry!! You Have Diabetes"

    return render_template('diabetes_predict.html', value=a)


# ---------------Heart Prediction-------------------------


@app.route('/heart')
def heart():
    return render_template('heart.html')


@app.route('/heart_predict', methods=['POST'])
def heart_predict():

    model = pickle.load(open('Models/heart.pkl', 'rb'))
    data_array = list()

    age = float(request.form['age'])
    gender = float(request.form['gender'])
    cp = float(request.form['cp'])
    trestbps = float(request.form['trestbps'])
    chol = float(request.form['chol'])
    fbs = float(request.form['fbs'])
    restecg = float(request.form['restecg'])
    thalach = float(request.form['thalach'])
    exang = float(random.randint(0, 1))
    oldpeak = float(request.form['oldpeak'])
    slope = float(random.randint(0, 2))
    ca = float(random.randint(0, 4))
    thal = float(request.form['thal'])

    data_array = data_array + [age, gender, cp, trestbps,
                               chol, fbs, restecg, thalach,exang, oldpeak, slope, ca, thal]

    data = np.array([data_array])
    value = int(model.predict(data))

    if value == 0:
        a = "Congratulations!! You Have Low Risk Of Heart Failure"
    else:
        a = "Sorry!! You Have High Risk Of Heart Failure"

    return render_template('heart_predict.html', value=a)


# ---------------Kidney Prediction-------------------------


@app.route('/kidney1')
def kidney1():
    return render_template('kidney_1.html')


@app.route('/kidney2', methods=['POST'])
def kidney2():

    if len(User.query.all()) > 0:
        for i in User.query.all():

            db.session.delete(i)
            db.session.commit()

        data = User(id=1,
                    age=float(request.form['age']),
                    bp=float(request.form['bp']),
                    sg=float(request.form['sg']),
                    al=float(random.randint(0, 5)),
                    su=float(random.randint(0, 5)),
                    rbc=float(request.form['rbc']),
                    pc=float(request.form['pc']),
                    pcc=float(request.form['pcc']),
                    ba=float(request.form['ba']),
                    bu=float(request.form['bu']),
                    sc=float(request.form['sc']),
                    sod=float(request.form['sod']))

        db.session.add(data)
        db.session.commit()

    else:

        data = User(id=1,
                    age=float(request.form['age']),
                    bp=float(request.form['bp']),
                    sg=float(request.form['sg']),
                    al=float(random.randint(0, 5)),
                    su=float(random.randint(0, 5)),
                    rbc=float(request.form['rbc']),
                    pc=float(request.form['pc']),
                    pcc=float(request.form['pcc']),
                    ba=float(request.form['ba']),
                    bu=float(request.form['bu']),
                    sc=float(request.form['sc']),
                    sod=float(request.form['sod']))

        db.session.add(data)
        db.session.commit()

    return render_template('kidney_2.html')


@app.route('/kidney_predict', methods=['POST'])
def kidney_predict():

    model = pickle.load(open('Models/kidney.pkl', 'rb'))
    data_array = list()

    # Form 1

    database = User.query.filter_by(id=1).first()

    age = database.age
    bp = database.bp
    sg = database.sg
    al = database.al
    su = database.su
    rbc = database.rbc
    pc = database.pc
    pcc = database.pcc
    ba = database.ba
    bu = database.bu
    sc = database.sc
    sod = database.sod

    # Form 2

    pot = float(request.form['pot'])
    hemo = float(request.form['hemo'])
    pcv = float(request.form['pcv'])
    wc = float(request.form['wc'])
    rc = float(request.form['rc'])
    htn = float(request.form['htn'])
    dm = float(request.form['dm'])
    cad = float(request.form['cad'])
    appet = float(request.form['appet'])
    pe = float(request.form['pe'])
    ane = float(random.randint(0, 2))

    data_array = data_array + [age, bp, sg, al, su, rbc, pc, pcc, ba, bu, sc, sod,
                               pot, hemo, pcv, wc, rc, htn, dm, cad, appet, pe, ane]

    data = np.array([data_array])
    value = int(model.predict(data))

    if value == 0:
        a = "Congratulations!! You Have Low Risk Of Kidney Failure"
    else:
        a = "Sorry!! You Have High Risk Of Kidney Failure"

    return render_template('kidney_predict.html', value=a)


# ---------------Liver Prediction-------------------------


@app.route('/liver')
def liver():
    return render_template('liver.html')


@app.route('/liver_predict', methods=['POST'])
def liver_predict():
    model = pickle.load(open('Models/liver.pkl', 'rb'))
    data_array = list()

    age = float(request.form['age'])
    gender = float(request.form['gender'])
    total_bilirubin = float(request.form['total_bilirubin'])
    alkaline_phosphotase = float(request.form['alkaline_phosphotase'])
    aat = float(request.form['aat'])
    total_protiens = float(request.form['total_protiens'])
    agr = float(request.form['agr'])

    data_array = data_array + [age, gender, total_bilirubin, alkaline_phosphotase,
                               aat, total_protiens, agr]

    data = np.array([data_array])
    value = int(model.predict(data))

    if value == 1:
        a = "Congratulations!! Your Liver Is Fine"
    else:
        a = "Sorry!! You Have High Risk Of Liver Failure"

    return render_template('liver_predict.html', value=a)


if __name__ == '__main__':
    app.run(debug=True)


