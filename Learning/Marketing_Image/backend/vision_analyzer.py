import os
import base64
import json
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.4,
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

system_prompt = """You are a professional marketing analyst and image describer.

Your task:
- Analyze the provided image in the context of the structured metadata.
- Provide a detailed, vivid description of the image that captures its marketing appeal.
- Focus on lighting, composition, and emotional tone.

Rules:
- Use ONLY information seen in the image or provided in the metadata.
- Output the description in plain text.
- Do NOT mention internal reasoning or JSON structures.
"""


prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", [
        {"type": "text", "text": "Image Metadata: {input_json}"},
        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,{image_base64}"}}
    ])
])

def get_image_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

image_path = "/home/md-tanvir-uddin-alif/Personal_Project/Gen_AI/Learning/Marketing_Image/images/IMG_3839.JPG"
input_data = {
    "campaign": {
        "event": "Social Media Reveal",
        "brand_tone": "vibrant and professional"
    },
    "user_intent": "Explain the visual elements for a catalog."
}

chain = prompt | llm | StrOutputParser()


final_description = chain.invoke({
    "input_json": json.dumps(input_data, indent=2),
    "image_base64": get_image_base64(image_path)
})

print("--- Marketing Description ---")
print(final_description)