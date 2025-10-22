# In api/index.py
import os
import json
from fastapi import FastAPI
# Import the correct StreamingResponse
from fastapi.responses import StreamingResponse
import google.generativeai as genai
import httpx

# --- Configuration ---
# This is the public URL for your "Workshop" server on Railway
# !! Replace this with your actual Railway URL !!
WORKSHOP_URL = "mcp-suite-production.up.railway.app"

# Set up the Google Gemini client
try:
    genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
except KeyError:
    print("ERROR: GOOGLE_API_KEY environment variable not set.")

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
                    parts.append({
                        "function_response": {
                            "name": item["name"],
                            "response": {"content": item["content"]}
                        }
                    })
                    role = "function"
                else:
                    parts.append({"text": str(item)})
            gemini_messages.append({"role": role, "parts": parts})
        else:
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
        model = genai.GenerativeModel('gemini-1.5-flash', tools=gemini_tools)
        gemini_messages = convert_messages_to_gemini(messages)
        
        # Call Gemini (non-streaming) to check for tool calls
        response = model.generate_content(gemini_messages)
        first_part = response.candidates[0].content.parts[0]

        # Check if Gemini wants to use a tool
        if first_part.function_call:
            # --- Tool Call Path ---
            
            function_call = first_part.function_call
            tool_name = function_call.name
            tool_params = dict(function_call.args)
            
            # Reformat tool name for our MCP server
            tool_name_mcp = tool_name.replace("_", ": ", 1)
            
            # Call your Railway "Workshop"
            tool_result = ""
            try:
                async with httpx.AsyncClient(timeout=30.0) as http_client:
                    tool_response = await http_client.post(
                        f"{WORKSHOP_URL}/call_tool", # We still need to add this endpoint to your server
                        json={"name": tool_name_mcp, "params": tool_params}
                    )
                    tool_response.raise_for_status()
                    tool_result = tool_response.json()
            except Exception as e:
                tool_result = {"error": f"Failed to call tool: {e}"}

            # Add the tool call and result to the history
            gemini_messages.append({"role": "model", "parts": [first_part]})
            gemini_messages.append({
                "role": "function",
                "parts": [{"function_response": {"name": tool_name, "response": {"content": json.dumps(tool_result)}}}]
            })
            
            # Call Gemini *again* with the tool result, this time streaming
            final_response_stream = model.generate_content(
                gemini_messages,
                stream=True
            )

            # This is the async generator that yields text chunks
            async def stream_generator():
                for chunk in final_response_stream:
                    if chunk.parts:
                        yield chunk.parts[0].text
            
            # Return the correct FastAPI StreamingResponse
            return StreamingResponse(stream_generator(), media_type="text/plain")

        else:
            # --- Simple Text Response Path ---
            # No tool was needed. Just stream the simple text response.
            
            # We must call generate_content *again* in stream=True mode
            text_response_stream = model.generate_content(
                gemini_messages,
                stream=True
            )

            async def text_stream_generator():
                for chunk in text_response_stream:
                    if chunk.parts:
                        yield chunk.parts[0].text
            
            return StreamingResponse(text_stream_generator(), media_type="text/plain")

    except Exception as e:
        print(f"An error occurred: {e}")
        return StreamingResponse(iter([f"An error occurred: {e}"]), media_type="text/plain")