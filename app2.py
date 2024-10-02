import os
import sys
import logging
from io import BytesIO
from pathlib import Path
from typing import List

from datetime import datetime

import chainlit as cl
from chainlit.config import config
from chainlit.element import Element

from openai import AsyncAssistantEventHandler, AsyncOpenAI, OpenAI

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Environment Variables
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
OPENAI_ASSISTANT_ID = os.environ.get("OPENAI_ASSISTANT_ID")
WHISPER_MODEL = os.environ.get("WHISPER_MODEL", "whisper-1")

# Validate Environment Variables
if not OPENAI_API_KEY:
    sys.exit("Error: OPENAI_API_KEY environment variable not set.")

if not OPENAI_ASSISTANT_ID:
    sys.exit("Error: OPENAI_ASSISTANT_ID environment variable not set.")

# Initialize OpenAI Clients
async_openai_client = AsyncOpenAI(api_key=OPENAI_API_KEY)
sync_openai_client = OpenAI(api_key=OPENAI_API_KEY)

# Retrieve Assistant
try:
    assistant = sync_openai_client.beta.assistants.retrieve(OPENAI_ASSISTANT_ID)
except Exception as e:
    logger.error("Failed to retrieve the assistant.")
    sys.exit("Error: Failed to retrieve the assistant.")

# Update Chainlit Config
config.ui.name = assistant.name

class EventHandler(AsyncAssistantEventHandler):
    """Custom event handler for processing assistant events."""

    def __init__(self, assistant_name: str) -> None:
        super().__init__()
        self.current_message: cl.Message = None
        self.current_step: cl.Step = None
        self.current_tool_call = None
        self.assistant_name = assistant_name

    async def on_text_created(self, text) -> None:
        """Handles the creation of new assistant text."""
        self.current_message = await cl.Message(author=self.assistant_name, content="").send()

    async def on_text_delta(self, delta, snapshot):
        """Streams incoming text delta to the client."""
        await self.current_message.stream_token(delta.value)

    async def on_text_done(self, text):
        """Updates the message when text streaming is done."""
        await self.current_message.update()

    async def on_tool_call_created(self, tool_call):
        """Handles the creation of a tool call."""
        self.current_tool_call = tool_call.id
        self.current_step = cl.Step(name=tool_call.type, type="tool")
        self.current_step.language = "python"
        self.current_step.created_at = datetime.utcnow()
        await self.current_step.send()

    async def on_tool_call_delta(self, delta, snapshot):
        """Processes deltas in tool calls, such as code interpreter outputs."""
        if snapshot.id != self.current_tool_call:
            self.current_tool_call = snapshot.id
            self.current_step = cl.Step(name=delta.type, type="tool")
            self.current_step.language = "python"
            self.current_step.start = datetime.utcnow()
            await self.current_step.send()

        if delta.type == "code_interpreter":
            if delta.code_interpreter.outputs:
                for output in delta.code_interpreter.outputs:
                    if output.type == "logs":
                        error_step = cl.Step(
                            name=delta.type,
                            type="tool",
                            is_error=True,
                            output=output.logs,
                            language="markdown",
                            start=self.current_step.start,
                            end=datetime.utcnow()
                        )
                        await error_step.send()
            elif delta.code_interpreter.input:
                await self.current_step.stream_token(delta.code_interpreter.input)

    async def on_tool_call_done(self, tool_call):
        """Updates the step when the tool call is complete."""
        self.current_step.end = datetime.utcnow()
        await self.current_step.update()

    async def on_image_file_done(self, image_file):
        """Handles incoming image files from the assistant."""
        image_id = image_file.file_id
        try:
            response = await async_openai_client.files.with_raw_response.content(image_id)
            image_element = cl.Image(
                name=image_id,
                content=response.content,
                display="inline",
                size="large"
            )
            if not self.current_message.elements:
                self.current_message.elements = []
            self.current_message.elements.append(image_element)
            await self.current_message.update()
        except Exception as e:
            logger.error(f"Failed to retrieve image file: {e}")
            await cl.Message(content="An error occurred while retrieving an image.").send()

@cl.step(type="tool")
async def speech_to_text(audio_buffer: BytesIO) -> str:
    """Transcribes audio to text using OpenAI's Whisper model."""
    audio_buffer.seek(0)  # Ensure the buffer is at the beginning
    try:
        response = await async_openai_client.audio.transcriptions.create(
            model=WHISPER_MODEL, file=audio_buffer
        )
        return response.text
    except Exception as e:
        logger.error(f"Error during speech to text transcription: {e}")
        raise e

async def upload_files(files: List[Element]) -> List[str]:
    """Uploads files to OpenAI and returns their IDs."""
    file_ids = []
    for file in files:
        try:
            uploaded_file = await async_openai_client.files.create(
                file=Path(file.path), purpose="assistants"
            )
            file_ids.append(uploaded_file.id)
            logger.info(f"Uploaded file {file.name} with ID {uploaded_file.id}")
        except Exception as e:
            logger.error(f"Error uploading file '{file.name}': {e}")
            await cl.Message(content=f"An error occurred while uploading file '{file.name}'.").send()
    return file_ids

def get_tools_for_file(file: Element) -> List[dict]:
    """Determine tools based on file type or other criteria."""
    # Placeholder logic for tool selection based on file type
    # Extend this function based on actual criteria
    return [{"type": "code_interpreter"}, {"type": "file_search"}]

async def process_files(files: List[Element]) -> List[dict]:
    """Processes files and prepares them for attachments."""
    file_ids = []
    if files:
        file_ids = await upload_files(files)

    attachments = [
        {
            "file_id": file_id,
            "tools": get_tools_for_file(file),
        }
        for file_id, file in zip(file_ids, files)
    ]
    return attachments

@cl.on_chat_start
async def start_chat():
    """Initializes a new chat session."""
    try:
        # Create a new thread in OpenAI
        thread = await async_openai_client.beta.threads.create()
        # Store the thread ID in the user session
        cl.user_session.set("thread_id", thread.id)
        logger.info(f"New chat started with thread ID {thread.id}")
        # Send the assistant's avatar and greeting message
        await cl.Avatar(name=assistant.name, path="./public/logo.png").send()
        await cl.Message(content=f"Hello, I'm {assistant.name}!", disable_feedback=True).send()
    except Exception as e:
        logger.error(f"Error starting chat: {e}")
        await cl.Message(content="An error occurred while starting the chat. Please try again later.").send()

@cl.on_message
async def main(message: cl.Message):
    """Processes user messages and interacts with the assistant."""
    thread_id = cl.user_session.get("thread_id")

    if not thread_id:
        await cl.Message(content="Session expired or invalid. Please refresh and start a new chat.").send()
        return

    # Process any attached files
    attachments = []
    if message.elements:
        attachments = await process_files(message.elements)

    try:
        # Add a message from the user to the thread
        await async_openai_client.beta.threads.messages.create(
            thread_id=thread_id,
            role="user",
            content=message.content,
            attachments=attachments,
        )
        logger.info(f"User message added to thread {thread_id}")

        # Create and stream a response from the assistant
        async with async_openai_client.beta.threads.runs.stream(
            thread_id=thread_id,
            assistant_id=assistant.id,
            event_handler=EventHandler(assistant_name=assistant.name),
        ) as stream:
            await stream.until_done()
            logger.info("Assistant response streamed successfully.")
    except Exception as e:
        logger.error(f"Error in main message handler: {e}")
        await cl.Message(content="An error occurred while processing your message. Please try again.").send()

@cl.on_audio_chunk
async def on_audio_chunk(chunk: cl.AudioChunk):
    """Handles streaming audio data from the user."""
    try:
        if chunk.isStart:
            buffer = BytesIO()
            # Set the buffer name based on the MIME type
            buffer.name = f"input_audio.{chunk.mimeType.split('/')[1]}"
            # Initialize the session for a new audio stream
            cl.user_session.set("audio_buffer", buffer)
            cl.user_session.set("audio_mime_type", chunk.mimeType)
            logger.info(f"Started receiving audio: {buffer.name}")

        # Write the chunk data to the buffer
        audio_buffer: BytesIO = cl.user_session.get("audio_buffer")
        audio_buffer.write(chunk.data)
    except Exception as e:
        logger.error(f"Error handling audio chunk: {e}")

@cl.on_audio_end
async def on_audio_end(elements: List[Element]):
    """Processes the complete audio message after the user stops speaking."""
    # Retrieve the audio buffer and MIME type from the session
    audio_buffer: BytesIO = cl.user_session.get("audio_buffer")
    audio_mime_type: str = cl.user_session.get("audio_mime_type")

    if not audio_buffer or not audio_mime_type:
        await cl.Message(content="No audio data received.").send()
        logger.warning("No audio data found in session.")
        return

    try:
        # Create an audio element for playback in the chat
        input_audio_el = cl.Audio(
            mime=audio_mime_type, content=audio_buffer.getvalue(), name=audio_buffer.name
        )
        await cl.Message(
            author="You",
            type="user_message",
            content="",
            elements=[input_audio_el, *elements],
        ).send()

        # Transcribe the audio
        transcription = await speech_to_text(audio_buffer)
        logger.info("Audio transcription completed.")

        # Clean up the audio buffer and MIME type from the session
        cl.user_session.delete("audio_buffer")
        cl.user_session.delete("audio_mime_type")
        audio_buffer.close()  # Close the buffer

        # Send the transcription as the user's message
        msg = cl.Message(author="You", content=transcription, elements=elements)
        await main(message=msg)
    except Exception as e:
        logger.error(f"Error processing audio: {e}")
        await cl.Message(content="An error occurred while processing your audio. Please try again.").send()
