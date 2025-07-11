from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

# Load the saved model
model = joblib.load('iris_model.pkl')

# Homepage: Show the input form
@app.route('/')
def home():
    return render_template('index.html')

# Prediction route: When user submits form
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input features
        features = [float(x) for x in request.form.values()]
        prediction = model.predict([features])[0]

        # Classes and corresponding image filenames
        classes = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
        images = ['setosa.jpg', 'versicolor.jpg', 'virginica.jpg']

        result = classes[prediction]
        image_file = images[prediction]

        return render_template('index.html',
                               prediction_text=f'🌸 Predicted Iris Species: {result}',
                               image_file=image_file)
    except:
        return render_template('index.html',
                               prediction_text="❌ Invalid input! Please enter valid numbers.")


if __name__ == '__main__':
    app.run(debug=True)
