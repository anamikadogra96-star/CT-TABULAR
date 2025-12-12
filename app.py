from flask import Flask, render_template, request, jsonify
from model import StudyPlannerModel
import os

app = Flask(__name__)
model = StudyPlannerModel()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.form
        gender = int(data['gender'])
        race = int(data['race'])
        parent_edu = int(data['parent_edu'])
        lunch = int(data['lunch'])
        test_prep = int(data['test_prep'])
        
        score, level = model.predict(gender, race, parent_edu, lunch, test_prep)
        timetable_data = model.generate_timetable(score)
        
        return render_template('result.html', 
                             score=round(score, 1),
                             level=level,
                             timetable=timetable_data['timetable'],
                             study_hours=timetable_data)
    except Exception as e:
        return f"Error: {str(e)}", 400

@app.route('/save_progress', methods=['POST'])
def save_progress():
    try:
        data = request.form
        math_h = float(data['math_hours'])
        reading_h = float(data['reading_hours'])
        writing_h = float(data['writing_hours'])
        
        progress = model.save_progress(math_h, reading_h, writing_h)
        return jsonify({
            'success': True,
            'progress': progress,
            'message': 'Progress saved successfully!'
        })
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 400

@app.route('/progress')
def progress():
    progress_data = model.get_progress_data()
    return render_template('progress.html', data=progress_data)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)

