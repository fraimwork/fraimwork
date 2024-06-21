from flask import Flask, render_template, request, redirect, url_for, flash
import pandas as pd
import os

app = Flask(__name__)
app.secret_key = 'supersecretkey'

CSV_FILE = './raw/samples.csv'

# Ensure the CSV file exists
if not os.path.exists(CSV_FILE):
    df = pd.DataFrame(columns=['dart', 'js'])
    df.to_csv(CSV_FILE, index=False)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/add_sample', methods=['POST'])
def add_sample():
    dart_code = request.form['dart']
    js_code = request.form['javascript']
    
    if dart_code and js_code:
        new_data = {'dart': dart_code, 'javascript': js_code}
        df = pd.read_csv(CSV_FILE)
        df = pd.concat([df, pd.DataFrame([new_data])], ignore_index=True)
        df.to_csv(CSV_FILE, index=False)
        flash('Sample added successfully!', 'success')
    else:
        flash('Both fields are required!', 'danger')
    
    return redirect(url_for('index'))

@app.route('/samples')
def samples():
    df = pd.read_csv(CSV_FILE)
    samples = df.to_dict(orient='records')
    return render_template('samples.html', samples=samples)

if __name__ == '__main__':
    app.run(debug=True, port=5100)