#!/bin/bash

# Check if a prompt is provided
if [ -z "$1" ]; then
  echo "Usage: $0 \"<your image prompt>\""
  exit 1
fi

PROMPT="$1"
SERVER_URL="http://localhost:10001"
A2A_TASK_ENDPOINT="/a2a/task"
IMAGE_ENDPOINT="/image"

echo "Sending prompt to A2A server: '$PROMPT'"

# Make the API call and store the response
RESPONSE=$(curl -s -X POST -H "Content-Type: application/json" -d "{\"input\": {\"prompt\": \"$PROMPT\"}}" "$SERVER_URL$A2A_TASK_ENDPOINT")

# Check if curl command was successful and response is not empty
if [ -z "$RESPONSE" ]; then
  echo "Error: No response from server. Is it running at $SERVER_URL?"
  exit 1
fi

# echo "Full server response: $RESPONSE"

# Extract the result message using jq
RESULT_MSG=$(echo "$RESPONSE" | jq -r '.result')

if [ "$RESULT_MSG" == "null" ] || [ -z "$RESULT_MSG" ]; then
  echo "Error: Could not parse 'result' from server response."
  echo "Server response: $RESPONSE"
  exit 1
fi

# Extract the image ID from the result message
# Assumes format: "Image generated successfully with ID: xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx"
IMAGE_ID=$(echo "$RESULT_MSG" | sed -n 's/.*ID: \([a-f0-9-]\{36\}\).*/\1/p')

if [ -z "$IMAGE_ID" ]; then
  echo "Error: Could not extract Image ID from result message: '$RESULT_MSG'"
  echo "Server response: $RESPONSE"
  exit 1
fi

IMAGE_URL="$SERVER_URL$IMAGE_ENDPOINT/$IMAGE_ID"

echo ""
echo "Image generation initiated."
echo "Image ID: $IMAGE_ID"
echo "Image URL: $IMAGE_URL"

# You can uncomment the line below to try and open the URL automatically on macOS
# open "$IMAGE_URL"
