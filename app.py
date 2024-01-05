from flask import request, render_template, send_file
from flask import Flask, redirect, url_for
import pandas as pd
import matplotlib.pyplot as plt
import os
from views import *

app = Flask(__name__)

models = [
    'Logistic Regression',
    'Support Vector Machine',
    'Decision Trees',
    'XGBoost',
    'Naive Bayes',
    'Random Forest',
    'Gradient Boosting Machines'
]


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/team')
def team():
    return render_template('team.html')


@app.route('/models', methods=['GET', 'POST'])
def model_selection():
    if request.method == 'POST':
        selected_model = request.form['model']
        return redirect(url_for('training', model=selected_model))
    return render_template('models.html', models=models)


@app.route('/training', methods=['POST'])
def training():
    if request.method == 'POST':
        selected_model = request.form['model']
        metrics = run_models(selected_model)
        data = {
            'model': selected_model,
            'accuracy': round(metrics.get('accuracy'),2),
            'precision': round(metrics.get('precision'),2),
            'recall': round(metrics.get('recall'),2),
            'f1_score': round(metrics.get('f1_score'),2)
        }
        return render_template('training.html', data=data)


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    prediction = None
    if request.method == 'POST':
        # import pdb;pdb.set_trace()
        model = 'random_forest'
        age_range = request.form['age_range']
        income_range = request.form['income_range']
        home_ownership = request.form['home_ownership']
        loan_intent = request.form['loan_intent']
        employment_length = request.form['employment_length']
        loan_grade = request.form['loan_grade']
        loan_range = request.form['loan_range']
        loan_int_rate = request.form['loan_int_rate']
        loan_default = request.form['loan_default']
        loan_income_percent = request.form['loan_income_percent']
        credit_hist_length = request.form['credit_hist_length']

        features = [age_range, income_range, home_ownership, loan_intent, employment_length, loan_grade, loan_range, loan_int_rate,
                    loan_default, loan_income_percent, credit_hist_length]
        features = [float(i) for i in features]
        print(features)

        prediction = get_predictions(features, model)  # Adjust according to your model's requirements

    # Render the HTML page with the prediction result
    return render_template('prediction.html', prediction=prediction, models=models)


@app.route('/upload', methods=['POST'])
def upload_file():
    file = request.files['file']
    filepath = os.path.join('files/', file.filename)
    file.save(filepath)
    return redirect(url_for('show_visualizations'))


@app.route('/show-visualizations', methods=['GET'])
def show_visualizations():
    run_eda()
    return render_template("visualizations.html")


@app.route('/graph')
def show_graph():
    # Code to read CSV, create graph, save graph as image
    return send_file('path/to/graph.png', mimetype='image/png')


if __name__ == '__main__':
    app.run(host="127.0.0.1", port=8000)

