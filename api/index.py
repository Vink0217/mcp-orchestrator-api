# In api/index.py
import os
import json
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
import google.generativeai as genai
import httpx

# --- Configuration ---
# This is the public URL for your "Workshop" server on Railway
# !! Replace this with your actual Railway URL if needed !!
WORKSHOP_URL = os.environ.get("WORKSHOP_URL", "https://vinayak-mcp-workshop.up.railway.app") # Read from env

# Set up the Google Gemini client
try:
    genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
except KeyError:
    print("ERROR: GOOGLE_API_KEY environment variable not set.")
    # You might want to raise an exception or handle this more robustly
    # For now, we'll let it potentially fail later if the key isn't found.

# Create the FastAPI app
app = FastAPI()

# --- Tool Definition (Gemini Format) ---
gemini_tools = [
    {
        "name": "FS_list_files",
        "description": "List all files in a given directory inside the sandbox.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "The directory path to list, e.g., '.'"}
            },
            "required": ["path"]
        }
    },
    {
        "name": "FS_read_file",
        "description": "Read the full content of a text file inside the sandbox.",
        "parameters": {
            "type":"object",
            "properties": {
                "path": {"type": "string", "description": "The file path to read, e.g., 'main.py'"}
            },
            "required": ["path"]
        }
    },
    {
        "name": "FS_write_file",
        "description": "Write text content to a file. Use overwrite=True to replace.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "The file path to write to."},
                "content": {"type": "string", "description": "The text content to write."},
                "overwrite": {"type": "boolean", "default": False}
            },
            "required": ["path", "content"]
        }
    }
    # ... You can add all your other FS and DB tools here ...
]

# --- Helper Function to Convert Messages ---
def convert_messages_to_gemini(messages):
    """Converts a list of Vercel AI SDK messages to Gemini's format."""
    gemini_messages = []
    for msg in messages:
        role = "model" if msg["role"] == "assistant" else "user"

        if isinstance(msg["content"], list):
            parts = []
            for item in msg["content"]:
                if item["type"] == "tool_use":
                    parts.append({
                        "function_call": {
                            "name": item["name"],
                            "args": item["input"]
                        }
                    })
                elif item["type"] == "tool_result":
                    # Ensure content is a string for Gemini
                    content_str = item["content"]
                    if not isinstance(content_str, str):
                        try:
                            content_str = json.dumps(content_str)
                        except TypeError:
                            content_str = str(content_str) # Fallback to string representation

                    parts.append({
                        "function_response": {
                            "name": item["name"], # Need to pass tool name back for Gemini
                            "response": {"content": content_str}
                        }
                    })
                    role = "function" # Gemini's role for tool results
                else:
                    parts.append({"text": str(item)}) # Fallback for text
            gemini_messages.append({"role": role, "parts": parts})
        else:
            # Standard text message
            gemini_messages.append({"role": role, "parts": [{"text": msg["content"]}]})

    return gemini_messages

# --- Chatbot Endpoint ---
@app.post("/api/chat")
async def chat(body: dict):
    """
    This is the main chatbot endpoint.
    It receives messages, calls Gemini, and uses tools.
    """
    messages = body.get('messages', [])

    try:
        # Initialize the model WITHOUT tools in constructor
        model = genai.GenerativeModel('gemini-1.5-flash')
        gemini_messages = convert_messages_to_gemini(messages)

        # Call Gemini (non-streaming) to check for tool calls, passing tools here
        response = model.generate_content(gemini_messages, tools=gemini_tools)

        # Check for potential errors or empty response
        if not response.candidates or not response.candidates[0].content or not response.candidates[0].content.parts:
             print("Warning: Gemini response was empty or malformed.")
             return StreamingResponse(iter(["Sorry, I received an unexpected response from the AI."]), media_type="text/plain")

        first_part = response.candidates[0].content.parts[0]

        # Check if Gemini wants to use a tool
        if hasattr(first_part, 'function_call') and first_part.function_call:
            # --- Tool Call Path ---

            function_call = first_part.function_call
            tool_name = function_call.name
            tool_params = dict(function_call.args)

            # Reformat tool name for our MCP server
            tool_name_mcp = tool_name.replace("_", ": ", 1)

            # --- Call your Railway "Workshop" ---
            tool_result = ""
            print(f"Attempting to call tool: {tool_name_mcp} with params: {tool_params}") # Debugging
            try:
                async with httpx.AsyncClient(timeout=30.0) as http_client:
                    tool_response = await http_client.post(
                        f"{WORKSHOP_URL}/call_tool", # Endpoint on your Railway server
                        json={"name": tool_name_mcp, "params": tool_params}
                    )
                    tool_response.raise_for_status() # Raise error for bad HTTP status
                    tool_result = tool_response.json()
                    print(f"Tool call successful. Result: {tool_result}") # Debugging
            except httpx.RequestError as e:
                tool_result = {"error": f"Network error calling tool: {e}"}
                print(f"Error calling tool (network): {e}") # Debugging
            except httpx.HTTPStatusError as e:
                tool_result = {"error": f"Tool server returned error {e.response.status_code}: {e.response.text}"}
                print(f"Error calling tool (HTTP status): {e}") # Debugging
            except Exception as e:
                tool_result = {"error": f"Unexpected error calling tool: {e}"}
                print(f"Error calling tool (unexpected): {e}") # Debugging

            # Prepare the tool result message for Gemini
            tool_result_content_str = json.dumps(tool_result) # Ensure it's a JSON string

            # Add the AI's tool call and our tool result to the history
            gemini_messages.append({"role": "model", "parts": [first_part]})
            gemini_messages.append({
                "role": "function",
                "parts": [{"function_response": {"name": tool_name, "response": {"content": tool_result_content_str}}}]
            })

            # Call Gemini *again* with the tool result, streaming the final answer, passing tools here
            final_response_stream = model.generate_content(
                gemini_messages,
                tools=gemini_tools, # Pass tools again
                stream=True
            )

            async def stream_generator():
                print("Streaming final response...") # Debugging
                async for chunk in final_response_stream:
                    if chunk.parts:
                        yield chunk.parts[0].text
                    # Handle potential errors during streaming
                    if hasattr(chunk, 'prompt_feedback') and chunk.prompt_feedback.block_reason:
                         print(f"Stream blocked: {chunk.prompt_feedback.block_reason}")
                         yield f"\n[Error: Response blocked - {chunk.prompt_feedback.block_reason}]"
                         break


            return StreamingResponse(stream_generator(), media_type="text/plain")

        else:
            # --- Simple Text Response Path ---
            print("No tool call detected, generating simple text response...") # Debugging
            # No tool was needed. Call again in stream=True mode, passing tools here
            text_response_stream = model.generate_content(
                gemini_messages,
                tools=gemini_tools, # Pass tools again
                stream=True
            )

            async def text_stream_generator():
                async for chunk in text_response_stream:
                    if chunk.parts:
                        yield chunk.parts[0].text
                     # Handle potential errors during streaming
                    if hasattr(chunk, 'prompt_feedback') and chunk.prompt_feedback.block_reason:
                         print(f"Stream blocked: {chunk.prompt_feedback.block_reason}")
                         yield f"\n[Error: Response blocked - {chunk.prompt_feedback.block_reason}]"
                         break

            return StreamingResponse(text_stream_generator(), media_type="text/plain")

    except Exception as e:
        print(f"An error occurred in chat endpoint: {e}") # Debugging
        # Return error as a stream
        async def error_stream():
            yield f"An error occurred: {e}"
        return StreamingResponse(error_stream(), media_type="text/plain", status_code=500)

@app.get("/")
def read_root():
    # Simple endpoint to check if the server is running
    return {"message": "Chatbot Brain is running. Use the /api/chat endpoint."}