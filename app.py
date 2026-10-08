from flask import Flask, render_template, request
import pickle
import pandas as pd

app = Flask(__name__)

# Load the AI model
print("Loading model...")
with open('phone_model.pkl', 'rb') as file:
    model = pickle.load(file)
print(" Model loaded!")


model_styles = {
    
    'iPhone 15 Pro Max': {'icon': 'fa-apple', 'color': '#333', 'bg': '#f4f4f4', 'brand': 'Apple'},
    'iPhone 15':         {'icon': 'fa-apple', 'color': '#333', 'bg': '#f4f4f4', 'brand': 'Apple'},
    'iPhone 13':         {'icon': 'fa-apple', 'color': '#333', 'bg': '#f4f4f4', 'brand': 'Apple'},
    'iPhone SE':         {'icon': 'fa-apple', 'color': '#333', 'bg': '#f4f4f4', 'brand': 'Apple'},
    
    
    'Galaxy S24 Ultra':  {'icon': 'fa-android', 'color': '#1428a0', 'bg': '#eef2ff', 'brand': 'Samsung'},
    'Galaxy S24':        {'icon': 'fa-android', 'color': '#1428a0', 'bg': '#eef2ff', 'brand': 'Samsung'},
    'Galaxy A54':        {'icon': 'fa-android', 'color': '#1428a0', 'bg': '#eef2ff', 'brand': 'Samsung'},
    'Galaxy A14':        {'icon': 'fa-android', 'color': '#1428a0', 'bg': '#eef2ff', 'brand': 'Samsung'}
}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        budget = int(request.form['budget'])
        storage = int(request.form['storage'])
        os_choice = int(request.form['os_choice'])
        camera = int(request.form['camera'])
        battery = int(request.form['battery'])
        gaming = int(request.form['gaming'])

        input_data = pd.DataFrame({
            'Budget': [budget],
            'Storage_GB': [storage],
            'OS_Choice': [os_choice],
            'Camera_Importance': [camera],
            'Battery_Importance': [battery],
            'Gaming_Importance': [gaming]
        })

        prediction = model.predict(input_data)[0]
        
        
        style = model_styles.get(prediction, {'icon': 'fa-mobile-screen', 'color': '#333', 'bg': '#eee', 'brand': 'Unknown'})

        return render_template('index.html', 
                               model_name=prediction, 
                               icon=style['icon'], 
                               color=style['color'], 
                               bg=style['bg'],
                               brand=style['brand'],
                               budget=budget)

    except Exception as e:
        return render_template('index.html', error=str(e))

if __name__ == '__main__':
    app.run(debug=True)
else:
    app = app #(Render.com)