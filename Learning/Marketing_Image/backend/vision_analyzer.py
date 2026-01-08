import os
import base64
import json
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.4,
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

# System prompt to convert description into structured metadata
metadata_system_prompt = """You are an expert at extracting structured image metadata for marketing purposes.

Your task:
- Analyze the provided image in the context of the structured metadata.
- Provide a detailed, vivid description of the image that captures its marketing appeal.
- Focus on lighting, composition, and emotional tone.

Rules:
- Use ONLY information seen in the image or provided in the metadata.
- Output the description in plain text.
- Do NOT mention internal reasoning or JSON structures.
"""

metadata_prompt = ChatPromptTemplate.from_messages([
    ("system", metadata_system_prompt),
    ("human", "{description}")
])

metadata_chain = metadata_prompt | llm | StrOutputParser()


def get_image_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def analyze_image(image_path, image_context=None):
    """
    1️⃣ Generate marketing description from image
    2️⃣ Convert description → structured metadata
    """
    system_prompt_desc = """You are a professional marketing analyst and image describer.

Task:
- Describe the provided image vividly with marketing appeal.
- Focus on lighting, composition, and emotional tone.
- Only use info visible in image or provided context.
- Output plain text.
"""
    prompt_desc = ChatPromptTemplate.from_messages([
        ("system", system_prompt_desc),
        ("human", [
            {"type": "text", "text": f"Image Context: {image_context}" if image_context else ""},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{get_image_base64(image_path)}"}}
        ])
    ])

    chain_desc = prompt_desc | llm | StrOutputParser()
    description = chain_desc.invoke({"input_json": json.dumps(image_context or {}, indent=2)})

    # Now extract structured metadata from description
    metadata_json = metadata_chain.invoke({"description": description})
    return metadata_json


data = analyze_image("/home/md-tanvir-uddin-alif/Personal_Project/Gen_AI/Learning/Marketing_Image/images/IMG_3839.JPG")

print(data)