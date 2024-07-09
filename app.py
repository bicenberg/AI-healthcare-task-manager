from pymongo import MongoClient
from flask import Flask, request, render_template, redirect, url_for, session
from healthcare import process_prompt, fillCollection, index_prompt
import json

app = Flask(__name__)
app.secret_key = 'supersecretkey'

# Load employees data
with open('employees.json', 'r') as file:
    employees = json.load(file)

# Initialize MongoDB client
client = MongoClient(open('mongodb_key.txt','r').read())
try:
    client.admin.command('ping')
except Exception as e:
    print(e)

OPEN_API_KEY = open('apikey.txt','r').read()

DB_NAME = 'healthcare'
db = client[DB_NAME]

# Clear existing collections
db.employees.delete_many({})    
db.hospital.delete_many({})
db.emp.delete_many({})
db.hsp.delete_many({})

# Fill collections and get necessary attributes
emp_attributes, hsp_attributes, d = fillCollection()

# Dictionary to keep track of tasks assigned to employees
tasks = {}

@app.route('/')
def index():
    history = session.get('history', [])
    return render_template('index.html', history=history)

@app.route('/process', methods=['POST'])
def process():
    prompt = request.form['prompt']
    emp_retriever, hsp_retriever = index_prompt(emp_attributes, hsp_attributes)
    
    names, task = process_prompt(prompt, emp_retriever, hsp_retriever, d)
    
    history = session.get('history', [])
    if len(history) >= 5:
        history.pop(0)
    if prompt not in history:
        history.append(prompt)
    session['history'] = history

    # Update the tasks dictionary
    for name in names:
        tasks[name] = task

    return render_template('result.html', task=task, history=history, names=names)

@app.route('/repeat/<prompt>')
def repeat(prompt):
    emp_retriever, hsp_retriever = index_prompt(emp_attributes, hsp_attributes)
    
    names, task = process_prompt(prompt, emp_retriever, hsp_retriever, d)
    
    history = session.get('history', [])
    if len(history) >= 5:
        history.pop(0)
    if prompt not in history:
        history.append(prompt)
    session['history'] = history

    # Update the tasks dictionary
    for name in names:
        tasks[name] = task

    return render_template('result.html', task=task, history=history, names=names)

@app.route('/tasks')
def tasks_view():
    unavailable_employees = {k: v for k, v in d.items() if v == 0}
    return render_template('tasks.html', tasks=tasks,history=session.get('history', []))

@app.route('/complete/<name>')
def complete(name):
    if name in d:
        d[name] = 1
    if name in tasks:
        del tasks[name]
    return redirect(url_for('tasks_view'))

@app.route('/employees')
def employees_page():
    return render_template('employees.html', employees=employees,
                           history=session.get('history', []))

if __name__ == '__main__':
    app.run(debug=True)
