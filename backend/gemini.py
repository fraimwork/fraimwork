import os
from google.cloud import aiplatform

def call_gemini_api(prompt):

    api_key = os.getenv("GEMINI_API_KEY")
    
    client = aiplatform.gapic.PredictionServiceClient()

    response = client.predict(
        endpoint="your-google-gemini-endpoint",
        instances=[{"prompt": prompt}],
        parameters={"max_tokens": 150, "temperature": 0.7},
    )

    broken_code = response.predictions[0]['text'].strip()
    return broken_code