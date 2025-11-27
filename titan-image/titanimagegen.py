import base64
import boto3
import json
import os

MODEL_ID = "amazon.titan-image-generator-v2:0"

bedrock = boto3.client("bedrock-runtime", region_name="us-west-2")

print("Enter a prompt:")
prompt = input("> ")

body = {
    "taskType": "TEXT_IMAGE",
  
    "textToImageParams": {
        "text": prompt 
    },
   
    "imageGenerationConfig": {
        "width": 1024,
        "height": 1024,
        "cfgScale": 8.0, 
        "seed": 0,
        "numberOfImages": 1
    }
}

response = bedrock.invoke_model(
    modelId=MODEL_ID,
    body=json.dumps(body)
)

result = json.loads(response["body"].read())


image_base64 = result["images"][0]

output_dir = "output"
os.makedirs(output_dir, exist_ok=True)

i = 1
while os.path.exists(os.path.join(output_dir, f"img_{i}.png")):
    i += 1

image_path = os.path.join(output_dir, f"img_{i}.png")

with open(image_path, "wb") as f:
    f.write(base64.b64decode(image_base64))

print(f"\nSaved: {image_path}")
