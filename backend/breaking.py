import random

from backend.gemini import call_gemini_api

def break_code_randomly(code):
    chunks = code.split('\n\n')  #figure out a better way to split idk

    chosen_chunk = random.choice(chunks)

    prompt = f"Please try to subtly introduce bugs to break this code: \n{chosen_chunk}"

    broken_chunk = call_gemini_api(prompt)

    broken_code = code.replace(chosen_chunk, broken_chunk)

    return broken_code