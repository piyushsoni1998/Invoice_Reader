import boto3, json

client = boto3.client("bedrock-runtime", region_name="us-east-1")

body = {
    "anthropic_version": "bedrock-2023-05-31",
    "messages": [
        {"role": "user", "content": "Reply with OK only"}
    ],
    "max_tokens": 10
}

response = client.invoke_model(
    modelId="anthropic.claude-3-sonnet-20240229-v1:0",
    body=json.dumps(body),
    contentType="application/json",
    accept="application/json"
)

print(json.loads(response["body"].read())["content"][0]["text"])
