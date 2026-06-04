import os
import json
from pathlib import Path
from typing import Optional, Literal
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

# Load environment variables cleanly
load_dotenv()

hf_token = (
    os.getenv("HF_TOKEN")
    or os.getenv("HUGGING_FACE_HUB_TOKEN")
    or os.getenv("HUGGINGFACEHUB_API_TOKEN")
    or os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")
)
if not hf_token:
    raise RuntimeError("Hugging Face API token not found. Set HF_TOKEN in your .env file.")

print("--- Connecting to Hugging Face Serverless Endpoint ---")

# 1. Initialize the Hugging Face base endpoint
llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation",
    huggingfacehub_api_token=hf_token,
    max_new_tokens=512,
    temperature=0.5
)

# 2. Wrap it in ChatHuggingFace
chat_model = ChatHuggingFace(llm=llm)

# 3. Define your Pydantic schema
class Review(BaseModel):
    key_themes: list[str] = Field(description="Write down all the key themes discussed in the review in a list")
    summary: str = Field(description="A brief summary of the review")
    sentiment: Literal["positive", "negative", "neutral"] = Field(
        description="Return sentiment of the review either negative, positive or neutral"
    )
    pros: Optional[list[str]] = Field(default=None, description="Write down all the pros inside a list")
    cons: Optional[list[str]] = Field(default=None, description="Write down all the cons inside a list")
    name: Optional[str] = Field(default=None, description="Write the name of the reviewer")

# FIX: Pass the raw JSON Schema instead of the Pydantic class to avoid NotImplementedError
structured_model = chat_model.with_structured_output(Review.model_json_schema())

# 4. Invoke the model
print("Sending request to Hugging Face...")
result = structured_model.invoke("""I recently upgraded to the Samsung Galaxy S24 Ultra, and I must say, it’s an absolute powerhouse! The Snapdragon 8 Gen 3 processor makes everything lightning fast—whether I’m gaming, multitasking, or editing photos. The 5000mAh battery easily lasts a full day even with heavy use, and the 45W fast charging is a lifesaver.

The S-Pen integration is a great touch for note-taking and quick sketches, though I don't use it often. What really blew me away is the 200MP camera—the night mode is stunning, capturing crisp, vibrant images even in low light. Zooming up to 100x actually works well for distant objects, but anything beyond 30x loses quality.

However, the weight and size make it a bit uncomfortable for one-handed use. Also, Samsung’s One UI still comes with bloatware—why do I need five different Samsung apps for things Google already provides? The $1,300 price tag is also a hard pill to swallow.

Pros:
Insanely powerful processor (great for gaming and productivity)
Stunning 200MP camera with incredible zoom capabilities
Long battery life with fast charging
S-Pen support is unique and useful
                                 
Review by Nitish Singh
""")

# Note: Because we passed the JSON schema, 'result' returns as a clean Python dictionary.
print(f"\nReviewer Name: {result.get('name')}\n")

print("Full Extracted Data:")
print(json.dumps(result, indent=2))