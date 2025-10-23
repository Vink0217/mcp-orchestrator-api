# In api/index.py
import os
import json
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
import google.generativeai as genai
# Corrected import for older library version
from google.ai import generativelanguage as glm
import httpx

# --- Configuration ---
# Read the Workshop URL from environment variables, with a fallback
WORKSHOP_URL = os.environ.get("WORKSHOP_URL", "https://vinayak-mcp-workshop.up.railway.app") # Read from env

# Set up the Google Gemini client
try:
    genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
except KeyError:
    print("ERROR: GOOGLE_API_KEY environment variable not set.")
    # Consider raising an exception or providing a clearer error response

# Create the FastAPI app
app = FastAPI()

# --- Tool Definition (Correct Gemini Format using glm) ---
gemini_tools = glm.Tool( # Use glm.Tool
    function_declarations=[
        glm.FunctionDeclaration( # Use glm.FunctionDeclaration
            name="FS_list_files",
            description="List all files in a given directory inside the sandbox.",
            parameters={
                "type_": glm.Type.OBJECT, # Use glm.Type.OBJECT
                "properties": {
                    "path": {"type_": glm.Type.STRING, "description": "The directory path to list, e.g., '.'"} # Use glm.Type.STRING
                },
                "required": ["path"]
            }
        ),
        glm.FunctionDeclaration( # Use glm.FunctionDeclaration
            name="FS_read_file",
            description="Read the full content of a text file inside the sandbox.",
            parameters={
                "type_": glm.Type.OBJECT, # Use glm.Type.OBJECT
                "properties": {
                    "path": {"type_": glm.Type.STRING, "description": "The file path to read, e.g., 'main.py'"} # Use glm.Type.STRING
                },
                "required": ["path"]
            }
        ),
        glm.FunctionDeclaration( # Use glm.FunctionDeclaration
            name="FS_write_file",
            description="Write text content to a file. Use overwrite=True to replace.",
            parameters={
                "type_": glm.Type.OBJECT, # Use glm.Type.OBJECT
                "properties": {
                    "path": {"type_": glm.Type.STRING, "description": "The file path to write to."}, # Use glm.Type.STRING
                    "content": {"type_": glm.Type.STRING, "description": "The text content to write."}, # Use glm.Type.STRING
                    "overwrite": {"type_": glm.Type.BOOLEAN, "description": "Default is False"} # Use glm.Type.BOOLEAN
                },
                "required": ["path", "content"]
            }
        )
        # ... Add declarations for your other tools here in the same format ...
    ]
)

# Gemini expects a list containing the Tool object
tools_list_for_gemini = [gemini_tools]

# --- Helper Function to Convert Messages ---
def convert_messages_to_gemini(messages):
    """Converts a list of Vercel AI SDK messages to Gemini's format."""
    gemini_messages = []
    for msg in messages:
        # Determine role, defaulting to 'user' if not specified
        role = msg.get("role", "user")
        if role == "assistant":
            role = "model"

        content = msg.get("content", "")

        # Handle different content structures (simple string vs. list of parts)
        if isinstance(content, list):
            parts = []
            for item in content:
                item_type = item.get("type", "text") # Default to text if type is missing
                if item_type == "tool_use":
                    parts.append({
                        "function_call": {
                            "name": item.get("name", ""),
                            "args": item.get("input", {})
                        }
                    })
                elif item_type == "tool_result":
                     # Ensure content is a string for Gemini
                    tool_content = item.get("content", "")
                    content_str = tool_content
                    if not isinstance(content_str, str):
                        try:
                            content_str = json.dumps(content_str)
                        except TypeError:
                            content_str = str(content_str) # Fallback

                    # In older versions, the 'name' for function_response should match the called function's name
                    tool_name_used = item.get("tool_use_id", item.get("name", "")) # Try to get the original name

                    parts.append({
                        "function_response": {
                            "name": tool_name_used,
                            "response": {"content": content_str}
                        }
                    })
                    role = "user" # Use 'user' role for function responses in this older version format
                else: # Assume text
                    parts.append({"text": str(item.get("text", str(item)))}) # Extract text or stringify
            gemini_messages.append({"role": role, "parts": parts})
        elif isinstance(content, str):
            # Standard text message
            gemini_messages.append({"role": role, "parts": [{"text": content}]})
        else:
             print(f"Warning: Skipping message with unknown content format: {content}")


    return gemini_messages

# --- Chatbot Endpoint ---
@app.post("/api/chat")
async def chat(body: dict):
    """
    Receives messages, calls Gemini, handles tool calls to the Workshop, and streams responses.
    """
    messages = body.get('messages', [])
    if not messages:
        async def empty_stream(): yield "No messages provided."
        return StreamingResponse(empty_stream(), media_type="text/plain", status_code=400)

    try:
        # Initialize the model WITHOUT tools in constructor
        model = genai.GenerativeModel('gemini-1.5-flash') # Or 'gemini-pro' if flash isn't available
        gemini_messages = convert_messages_to_gemini(messages)

        # Call Gemini (non-streaming) to check for tool calls, passing tools list
        print("Calling Gemini (check for tool calls)...") # Debug
        response = model.generate_content(gemini_messages, tools=tools_list_for_gemini)

        # Safety check for response structure
        if not response.candidates or not response.candidates[0].content or not response.candidates[0].content.parts:
             print(f"Warning: Gemini response invalid. Response: {response}") # Debug
             async def error_stream(): yield "Sorry, I received an unexpected response from the AI."
             return StreamingResponse(error_stream(), media_type="text/plain", status_code=500)

        first_part = response.candidates[0].content.parts[0]

        # Check if Gemini wants to use a tool (using older attribute check)
        if hasattr(first_part, 'function_call') and first_part.function_call and first_part.function_call.name:
            # --- Tool Call Path ---
            print("Gemini requested tool call.") # Debug

            function_call = first_part.function_call
            tool_name = function_call.name
            tool_params = dict(function_call.args)

            # Reformat tool name for our MCP server (e.g., "FS_list_files" -> "FS: list_files")
            tool_name_mcp = tool_name.replace("_", ": ", 1)

            # --- Call your Railway "Workshop" ---
            tool_result = ""
            print(f"Attempting to call tool on Workshop: {tool_name_mcp} with params: {tool_params}") # Debug
            try:
                async with httpx.AsyncClient(timeout=30.0) as http_client:
                    tool_response = await http_client.post(
                        f"{WORKSHOP_URL}/call_tool", # Endpoint on your Railway server
                        json={"name": tool_name_mcp, "params": tool_params}
                    )
                    tool_response.raise_for_status() # Raise error for bad HTTP status (4xx or 5xx)
                    tool_result = tool_response.json()
                    print(f"Tool call successful. Result: {tool_result}") # Debug
            except httpx.RequestError as e:
                tool_result = {"error": f"Network error contacting Workshop: {e}"}
                print(f"Error calling tool (network): {e}") # Debug
            except httpx.HTTPStatusError as e:
                tool_result = {"error": f"Workshop server returned error {e.response.status_code}: {e.response.text}"}
                print(f"Error calling tool (HTTP status {e.response.status_code}): {e.response.text}") # Debug
            except Exception as e:
                tool_result = {"error": f"Unexpected error calling tool: {e}"}
                print(f"Error calling tool (unexpected): {e}") # Debug

            # Prepare the tool result message for Gemini
            tool_result_content_str = json.dumps(tool_result) # Ensure it's a JSON string

            # Add the AI's tool call and our tool result to the history
            gemini_messages.append({"role": "model", "parts": [first_part]}) # Append AI's request
            gemini_messages.append({                             # Append our result
                "role": "user", # Use 'user' role for function response in older format
                "parts": [{"function_response": {"name": tool_name, "response": {"content": tool_result_content_str}}}]
            })

            # Call Gemini *again* with the tool result, streaming the final answer, passing tools list
            print("Calling Gemini again with tool result...") # Debug
            final_response_stream = model.generate_content(
                gemini_messages,
                tools=tools_list_for_gemini, # Pass tools again
                stream=True
            )

            async def stream_generator():
                print("Streaming final response...") # Debug
                try:
                    async for chunk in final_response_stream:
                        # Check for text parts within the chunk
                        if chunk.parts:
                           for part in chunk.parts:
                               if hasattr(part, 'text') and part.text:
                                    yield part.text
                        # Handle potential safety blocks during streaming
                        if hasattr(chunk, 'prompt_feedback') and hasattr(chunk.prompt_feedback, 'block_reason') and chunk.prompt_feedback.block_reason:
                             print(f"Stream blocked: {chunk.prompt_feedback.block_reason}") # Debug
                             yield f"\n[Error: Response blocked due to {chunk.prompt_feedback.block_reason_message or chunk.prompt_feedback.block_reason}]"
                             break
                except Exception as stream_error:
                    print(f"Error during final response streaming: {stream_error}") # Debug
                    yield f"\n[Error streaming final response: {stream_error}]"

            return StreamingResponse(stream_generator(), media_type="text/plain")

        else:
            # --- Simple Text Response Path ---
            print("No tool call detected, generating simple text response...") # Debug
            # No tool was needed. Call again in stream=True mode, passing tools list
            text_response_stream = model.generate_content(
                gemini_messages,
                tools=tools_list_for_gemini, # Pass tools again
                stream=True
            )

            async def text_stream_generator():
                print("Streaming simple text response...") # Debug
                try:
                    async for chunk in text_response_stream:
                         # Check for text parts within the chunk
                        if chunk.parts:
                           for part in chunk.parts:
                               if hasattr(part, 'text') and part.text:
                                    yield part.text
                        # Handle potential safety blocks during streaming
                        if hasattr(chunk, 'prompt_feedback') and hasattr(chunk.prompt_feedback, 'block_reason') and chunk.prompt_feedback.block_reason:
                             print(f"Stream blocked: {chunk.prompt_feedback.block_reason}") # Debug
                             yield f"\n[Error: Response blocked due to {chunk.prompt_feedback.block_reason_message or chunk.prompt_feedback.block_reason}]"
                             break
                except Exception as stream_error:
                    print(f"Error during simple text streaming: {stream_error}") # Debug
                    yield f"\n[Error streaming text response: {stream_error}]"

            return StreamingResponse(text_stream_generator(), media_type="text/plain")

    except Exception as e:
        print(f"An error occurred in chat endpoint: {e}") # Debug
        # Return error as a stream
        async def error_stream():
            yield f"An error occurred: {e}"
        return StreamingResponse(error_stream(), media_type="text/plain", status_code=500)

@app.get("/")
def read_root():
    # Simple endpoint to check if the server is running
    return {"message": "Chatbot Brain is running. Use the /api/chat endpoint."}