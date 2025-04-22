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

    # Properly encode inputs and extract attention mask
    inputs = tokenizer(themed_prompt, return_tensors="pt", padding=True, truncation=True)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    # Generate the story
    output = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_length=length + input_ids.shape[1],
        do_sample=True,
        num_return_sequences=1,
        no_repeat_ngram_size=2,
        temperature=0.7,
        top_k=50,
        top_p=0.95,
        pad_token_id=tokenizer.eos_token_id
    )

    # Decode generated output
    generated_texts = []
    for generated_sequence in output:
        generated_text = tokenizer.decode(generated_sequence, skip_special_tokens=True)
        if themed_prompt in generated_text:
            generated_text = generated_text.replace(themed_prompt, "", 1).strip()
        generated_texts.append(generated_text)

    return jsonify({"story": generated_texts})

if __name__ == '__main__':
    app.run(debug=True, port=8000)
