from flask import Flask, request, jsonify, render_template
import numpy as np
import tensorflow as tf
import pickle

app = Flask(__name__)

# Load your trained model and vocabulary
model = tf.keras.models.load_model('character_mlp_model.h5')
with open('vocabulary.pkl', 'rb') as f:
    vocab = pickle.load(f)
stoi, itos = vocab['stoi'], vocab['itos']
context_length = 3  # This should match your training setting

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate():
    partial_input = request.form.get('input_text', '').lower()
    max_suggestions = int(request.form.get('max_suggestions', 5))
    
    # Prepare context for model input (pad or truncate to context_length)
    partial_indices = [stoi.get(ch, 0) for ch in partial_input]
    if len(partial_indices) > context_length:
        context = partial_indices[-context_length:]
    else:
        context = [0] * (context_length - len(partial_indices)) + partial_indices
    
    suggestions = []
    for _ in range(max_suggestions):
        generated = list(partial_input)
        context_copy = context.copy()
        for _ in range(20):  # limit max generation length
            x = np.array([context_copy])
            probs = model.predict(x, verbose=0)[0]
            next_idx = np.random.choice(len(probs), p=probs)
            if next_idx == 0:
                break
            next_char = itos[next_idx]
            generated.append(next_char)
            context_copy = context_copy[1:] + [next_idx]
        suggestions.append(''.join(generated))

    return jsonify({'suggestions': suggestions})

if __name__ == '__main__':
    app.run(debug=True)
