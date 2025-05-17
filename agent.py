"""
CrewAI Agent with A2A Protocol for image generation using OpenAI DALL-E
"""

import base64
import io
import json
import os
import uuid
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Type

import openai
from openai import OpenAI
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
from PIL import Image
import requests
import re

from crewai import Agent, Task, Crew, Process
from crewai.tools import BaseTool

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Ensure OPENAI_API_KEY is set for CrewAI default LLM configuration
if not os.getenv('OPENAI_API_KEY'):
    logger.warning("OPENAI_API_KEY environment variable not set. CrewAI may not function correctly.")
# Optionally set a default model for CrewAI agents if not specified per agent
# os.environ["OPENAI_MODEL_NAME"] = "gpt-4o-mini"

# Image cache to store generated images
IMAGE_CACHE: Dict[str, bytes] = {}

# --- Core Image Generation Functions (used by the CrewAI tool) ---
def generate_image_with_openai(client: OpenAI, prompt: str, cache_dir: Optional[str] = None) -> tuple[str, Optional[str]]:
    """
    Generate an image based on the given prompt using OpenAI DALL-E.
    Returns tuple of (result_message, image_id or None).
    """
    try:
        response = client.images.generate(
            model="dall-e-3",
            prompt=prompt,
            size="1024x1024",
            quality="standard",
            n=1,
            response_format="url"  # Request URL to download the image
        )
        
        image_url = response.data[0].url
        logger.info(f"OpenAI response URL: {image_url}")
        if not image_url:
            logger.error("Failed to get image URL from OpenAI response.")
            return "Failed to get image URL from OpenAI response.", None

        logger.info(f"Attempting to download image from: {image_url}")
        image_response = requests.get(image_url)
        image_response.raise_for_status()
        image_bytes = image_response.content
        logger.info(f"Successfully downloaded {len(image_bytes)} bytes from {image_url}")
        
        logger.info("Verifying downloaded image...")
        img = Image.open(io.BytesIO(image_bytes))
        img.verify()
        logger.info("Image verified successfully.")
        
        image_id = str(uuid.uuid4())
        IMAGE_CACHE[image_id] = image_bytes
        logger.info(f"Image cached with ID: {image_id}")
        
        if cache_dir:
            cache_path = Path(cache_dir)
            cache_path.mkdir(exist_ok=True)
            with open(cache_path / f"{image_id}.png", "wb") as f:
                f.write(image_bytes)
            logger.info(f"Image saved to cache directory: {cache_path / f'{image_id}.png'}")
        
        logger.info(f"Image generation process completed for ID: {image_id}")
        return f"Image generated successfully with ID: {image_id}", image_id
            
    except openai.APIError as e:
        logger.error(f"OpenAI API error generating image: {e}")
        return f"OpenAI API error: {e}", None
    except requests.RequestException as e:
        logger.error(f"Error downloading image from OpenAI: {e}")
        return f"Failed to download image: {e}", None
    except Exception as e:
        logger.error(f"Error generating image with OpenAI: {e}")
        return f"Failed to generate image: {e}", None

def edit_image_with_openai(client: OpenAI, image_id_to_edit: Optional[str], edit_prompt: str, cache_dir: Optional[str] = None) -> tuple[str, Optional[str]]:
    """
    "Edits" an image by generating a new one based on the edit_prompt using OpenAI DALL-E.
    The image_id_to_edit is for context/logging; DALL-E 3 regenerates the image.
    """
    if image_id_to_edit and image_id_to_edit not in IMAGE_CACHE:
        logger.warning(f"Image ID {image_id_to_edit} for editing not found in cache. Proceeding with generation based on prompt alone.")
    
    logger.info(f"Attempting 'edit' (regeneration) for image ID '{image_id_to_edit if image_id_to_edit else 'N/A'}' with prompt: {edit_prompt}")
    return generate_image_with_openai(client, edit_prompt, cache_dir)

def extract_image_id(text: str) -> Optional[str]:
    """Extract image ID from a text."""
    if not text: 
        return None
    match = re.search(r'ID: ([a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12})', text, re.IGNORECASE)
    if match:
        return match.group(1)
    uuids = re.findall(r'([a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12})', text, re.IGNORECASE)
    if uuids:
        return uuids[0]
    return None

# --- CrewAI Tool Definition ---
class OpenAIImageGenerationToolInput(BaseModel):
    prompt: str = Field(description="The textual prompt for image generation or editing instructions.")
    image_id_to_edit: Optional[str] = Field(default=None, description="Optional. The ID of the image to be 'edited'. For DALL-E 3, this means regenerating the image based on the prompt.")

class OpenAIImageGenerationTool(BaseTool):
    name: str = "OpenAI Image Generation Tool"
    description: str = "Generates a new image or 'edits' an existing one by regeneration using OpenAI DALL-E based on a textual prompt. For editing, provide the original image ID and edit instructions in the prompt."
    args_schema: Type[BaseModel] = OpenAIImageGenerationToolInput
    client: OpenAI = None
    cache_dir: Optional[str] = None

    def __init__(self, cache_dir: Optional[str] = None, **kwargs):
        super().__init__(**kwargs)
        self.client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        if not os.getenv('OPENAI_API_KEY'):
             raise ValueError("OPENAI_API_KEY not set, tool cannot function.")
        self.cache_dir = cache_dir

    def _run(self, prompt: str, image_id_to_edit: Optional[str] = None) -> str:
        logger.info(f"OpenAIImageGenerationTool received prompt: '{prompt}', image_id_to_edit: '{image_id_to_edit}'")
        if image_id_to_edit:
            # If an image_id_to_edit is provided, it's an 'edit' task.
            # The prompt should contain the actual editing instructions.
            result_msg, _ = edit_image_with_openai(
                client=self.client, 
                image_id_to_edit=image_id_to_edit, 
                edit_prompt=prompt, # The main prompt is used as edit instructions
                cache_dir=self.cache_dir
            )
        else:
            # New image generation
            result_msg, _ = generate_image_with_openai(
                client=self.client, 
                prompt=prompt, 
                cache_dir=self.cache_dir
            )
        return result_msg

# --- Pydantic models for A2A API ---
class A2ATaskInput(BaseModel):
    prompt: str

class A2ATask(BaseModel):
    input: A2ATaskInput

class A2AArtifact(BaseModel):
    type: str
    data: str

class A2AResponse(BaseModel):
    result: str
    artifacts: List[A2AArtifact]

# --- FastAPI Server Setup ---
def setup_a2a_server(cache_dir: Optional[str] = None) -> FastAPI:
    """Set up and return the A2A server with CrewAI-based image generation capability."""
    app = FastAPI()

    # Initialize the CrewAI tool
    shared_image_tool = OpenAIImageGenerationTool(cache_dir=cache_dir)

    @app.post("/a2a/task")
    async def handle_task(task_input: A2ATask) -> A2AResponse:
        """Handle incoming A2A tasks using CrewAI."""
        user_prompt = task_input.input.prompt
        if not user_prompt:
            raise HTTPException(status_code=400, detail="Prompt is required")

        logger.info(f"Received A2A task with prompt: {user_prompt}")

        # Check if this is an edit request
        is_edit = "edit" in user_prompt.lower() and "image_id" in user_prompt.lower()
        image_id_to_edit = None
        
        if is_edit:
            # Try to extract the image ID for editing
            match = re.search(r'image_id\s*[:=]\s*([a-f0-9-]+)', user_prompt, re.IGNORECASE)
            if match:
                image_id_to_edit = match.group(1)
                logger.info(f"Extracted image_id_to_edit: {image_id_to_edit}")

        # Define CrewAI Agent
        visual_creator_agent = Agent(
            role='Visual Creator AI',
            goal='Generate stunning and accurate images from textual descriptions, or create new images based on edit requests, using the OpenAI DALL-E tool.',
            backstory='An advanced AI specializing in digital art and image synthesis. It meticulously translates prompts into visual realities or creatively reinterprets existing concepts for modifications.',
            tools=[shared_image_tool],
            verbose=True,
            allow_delegation=False
            # LLM will use OpenAI by default if OPENAI_API_KEY is set.
        )

        # Create CrewAI Task
        generation_task = Task(
            description=f"Process the following user request for image generation or editing: '{user_prompt}'. "
                        f"If the request mentions editing an image and includes an 'image_id:' field, treat it as an edit request. "
                        f"Use the OpenAI Image Generation Tool to create or edit the image. "
                        f"For editing, provide both the image_id_to_edit parameter and the editing instructions as the prompt parameter.",
            expected_output="A string message indicating the result of the image generation. If successful, it must include 'Image generated successfully with ID: <UUID>'. If failed, it must provide an error message.",
            agent=visual_creator_agent
        )

        # Create and run CrewAI Crew
        crew = Crew(
            agents=[visual_creator_agent],
            tasks=[generation_task],
            process=Process.sequential,
            verbose=True
        )

        # Run the crew and get the result
        logger.info("Starting CrewAI process for image generation...")
        crew_result = crew.kickoff()
        
        # Convert CrewOutput to string for processing
        crew_result_str = str(crew_result)
        logger.info(f"CrewAI process completed with result: {crew_result_str}")

        # Extract image_id from the result
        image_id = extract_image_id(crew_result_str)
        logger.info(f"Extracted image_id from result: {image_id}")

        if not image_id or image_id not in IMAGE_CACHE:
            logger.warning(f"No valid image_id found in result or image not in cache. image_id={image_id}")
            return A2AResponse(result=crew_result, artifacts=[])
        
        # Create artifact with the image
        image_bytes = IMAGE_CACHE[image_id]
        b64_image = base64.b64encode(image_bytes).decode('utf-8')
        
        return A2AResponse(
            result=crew_result_str,  # Use the string representation
            artifacts=[
                A2AArtifact(type="image/png", data=b64_image)
            ]
        )
    
    # Add a route to get a specific image by ID
    @app.get("/image/{image_id}")
    async def get_image(image_id: str):
        """Get an image by its ID."""
        if image_id not in IMAGE_CACHE:
            raise HTTPException(status_code=404, detail="Image not found")
        
        image_bytes = IMAGE_CACHE[image_id]
        # Return as a streaming response for direct image viewing in browser if desired
        return StreamingResponse(io.BytesIO(image_bytes), media_type="image/png")
    
    # Add the health check endpoint
    @app.get("/a2a/healthz")
    async def health_check():
        return {"status": "ok"}
    
    # Add the metadata endpoint
    @app.get("/a2a/metadata")
    async def get_metadata():
        return {
            "name": "CrewAI OpenAI Image Generator Agent",
            "version": "1.2.0",
            "description": "An image generation agent using CrewAI and OpenAI DALL-E 3",
            "author": "A2A CrewAI Integration",
            "capabilities": ["image-generation", "image-edit-via-regeneration"]
        }
    
    return app
