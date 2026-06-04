import os
import json
from pathlib import Path
from typing import TypedDict, Annotated, Optional, Literal
from dotenv import load_dotenv
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

# 1. Load environment variables cleanly
# Looks for .env in the parent folder or current folder fallback
env_path = Path(__file__).resolve().parents[1] / ".env"
if env_path.exists():
    load_dotenv(dotenv_path=env_path)
else:
    load_dotenv()

# Retrieve the token
hf_token = (
    os.getenv("HF_TOKEN")
    or os.getenv("HUGGING_FACE_HUB_TOKEN")
    or os.getenv("HUGGINGFACEHUB_API_TOKEN")
    or os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")
)
if not hf_token:
    raise RuntimeError("Hugging Face API token not found. Set HF_TOKEN in your .env file.")

print("--- Connecting to Hugging Face Shared Serverless Router ---")

# 2. Initialize the base HuggingFace Endpoint
llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation",
    huggingfacehub_api_token=hf_token,
    max_new_tokens=512,
    temperature=0.5
)

# 3. Wrap it in ChatHuggingFace to enable structured output & chat formatting
chat_model = ChatHuggingFace(llm=llm)

# schema
class Review(TypedDict):
    key_themes: Annotated[list[str], "Write down all the key themes discussed in the review in a list"]
    summary: Annotated[str, "A brief summary of the review"]
    sentiment: Annotated[Literal["positive", "negative", "neutral"], "Return sentiment of the review"]
    pros: Annotated[Optional[list[str]], "Write down all the pros inside a list"]
    cons: Annotated[Optional[list[str]], "Write down all the cons inside a list"]
    name: Annotated[Optional[str], "Write the name of the reviewer"]

# 4. Bind the schema to the chat model
structured_model = chat_model.with_structured_output(Review)

# Sample text to analyze
sample_review = """I recently upgraded to the Samsung Galaxy S24 Ultra, and I must say, it’s an absolute powerhouse! The Snapdragon 8 Gen 3 processor makes everything lightning fast—whether I’m gaming, multitasking, or editing photos. The 5000mAh battery easily lasts a full day even with heavy use, and the 45W fast charging is a lifesaver.

The S-Pen integration is a great touch for note-taking and quick sketches, though I don't use it often. What really blew me away is the 200MP camera—the night mode is stunning, capturing crisp, vibrant images even in low light. Zooming up to 100x actually works well for distant objects, but anything beyond 30x loses quality.

However, the weight and size make it a bit uncomfortable for one-handed use. Also, Samsung’s One UI still comes with bloatware—why do I need five different Samsung apps for things Google already provides? The $1,300 price tag is also a hard pill to swallow.

Pros:
Insanely powerful processor (great for gaming and productivity)
Stunning 200MP camera with incredible zoom capabilities
Long battery life with fast charging
S-Pen support is unique and useful
                                 
Review by Amit Yadav
"""

print("Sending request to Llama-3.1 via Hugging Face...")

# Invoke the structured model
result = structured_model.invoke(sample_review)

# Print just the name
print(f"\nReviewer Name: {result.get('name')}\n")

# Print the entire structured response cleanly to inspect it
print("Full Extracted Data:")
print(json.dumps(result, indent=2))