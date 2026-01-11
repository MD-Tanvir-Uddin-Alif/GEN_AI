from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import json
import os


from vision_analyzer import analyze_image

load_dotenv()



llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.4,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)

# llm = ChatGoogleGenerativeAI(
#     model="gemini-2.5-flash",
#     temperature=0.4,
#     google_api_key=os.getenv("GOOGLE_API_KEY")
# )


system_prompt = """You are a professional AI prompt engineer for image generation systems.

Your task:
- Convert structured image metadata and user intent into ONE high-quality image generation prompt.
- The output must be optimized for image generation models (Stable Diffusion / SDXL).

Rules:
- Do NOT invent products, brands, logos, or colors.
- Use ONLY the information provided.
- If user intent is vague, improve it intelligently.
- If user intent is detailed, refine it without changing meaning.
- Do NOT mention internal reasoning.
- Do NOT output JSON, markdown, or explanations.
- Output ONE single prompt in plain text.

Focus on:
- Visual clarity
- Composition
- Lighting
- Style consistency
- Marketing quality

"""

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input_json}")
])


image_path = "/home/md-tanvir-uddin-alif/Personal_Project/Gen_AI/Learning/Marketing_Image/images/IMG_3839.JPG"
image_metadata = analyze_image(image_path)

input_data = {
  "image_metadata": image_metadata,
  "user_prompt": "i want to sale my e-commerce product",
  "campaign": {
    "event": "sale",
    "discount": "30%",
    "platform": "Facebook banner",
    "brand_tone": "energetic and modern"
  }
}


chain = prompt | llm | StrOutputParser()

final_prompt = chain.invoke({
    "input_json": json.dumps(input_data, ensure_ascii=False, indent=2)
})


print(final_prompt)



