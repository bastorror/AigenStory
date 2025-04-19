from flask import Flask, request, jsonify, render_template
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import os

app = Flask(__name__)

# Load your custom model
tokenizer = GPT2Tokenizer.from_pretrained('my_custom_model')
model = GPT2LMHeadModel.from_pretrained('my_custom_model')

# You can customize these themes to modify story generation
theme_dict = {
    'fantasy': ['dragon', 'castle', 'wizard', 'knight'],
    'sci-fi': ['spaceship', 'alien', 'robot', 'future'],
    'mystery': ['detective', 'clue', 'murder', 'suspense'],
    'romance': ['love', 'heart', 'kiss', 'passion'],
    'adventure': ['treasure', 'explore', 'journey', 'danger']
}

@app.route('/')
def index():
    return render_template('index.html')  # Ensure index.html is in the "templates" folder

@app.route("/generate", methods=["POST"])
def generate_text():
    data = request.get_json()
    prompt = data.get("prompt", "")
    length = int(data.get("length", 150))  # default to 150 if not provided
    theme = data.get("theme", "adventure")  # default to 'adventure' theme if not provided
    
    # Modify the prompt based on the selected theme
    theme_keywords = theme_dict.get(theme, ['adventure'])
    theme_intro = f"Once upon a time in a {theme_keywords[0]} setting, "
    themed_prompt = theme_intro + prompt

    # Encode the prompt with the theme modification
    input_ids = tokenizer.encode(themed_prompt, return_tensors="pt")
    
    # Generate the story using the GPT2 model
    output = model.generate(
        input_ids,
        max_length=length + len(input_ids[0]),
        do_sample=True,
        temperature=0.9,
        top_k=50,
        top_p=0.95
    )
    
    # Decode the generated output
    story = tokenizer.decode(output[0], skip_special_tokens=True)

    # Remove the theme intro part from the story output
    generated_story = story[len(themed_prompt):].strip()

    return jsonify({"story": generated_story})

if __name__ == '__main__':
    app.run(debug=True, port=8000)
