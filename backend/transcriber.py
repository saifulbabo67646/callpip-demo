import asyncio
import logging

from typing import Annotated

from dotenv import load_dotenv
from livekit import rtc
from livekit.agents import (
    AutoSubscribe,
    JobContext,
    WorkerOptions,
    cli,
    stt,
    transcription,
)
from livekit.agents.llm import (
    ChatContext,
    ChatMessage,
    FunctionContext,
    TypeInfo,
    ai_callable,
)
from livekit.plugins import openai, silero

load_dotenv(dotenv_path=".env.local")

logger = logging.getLogger("transcriber")

# Initialize chat context for LLM
chat_context = ChatContext(
    messages=[
        ChatMessage(
            role="system",
            content=(
                "You are a helpful assistant that responds to questions and queries. "
                "Keep your responses concise and relevant."
            ),
        )
    ]
)

class QuestionDetectionFunction(FunctionContext):
    """This class defines functions that handle question and intent detection."""

    @ai_callable(
        description="Called when the text appears to be a question or when the user is seeking information"
    )
    async def detect_question(
        self,
        text: Annotated[
            str,
            TypeInfo(
                description="The text to analyze for question/intent detection"
            ),
        ],
    ) -> bool:
        """Returns True if the text is a question or seeks information."""
        return True

async def _process_with_llm(text: str, room: rtc.Room, stt_forwarder: transcription.STTSegmentsForwarder):
    """Process the transcribed text with LLM and send response as text"""
    try:
        logger.info(f"Processing text with LLM: {text}")
        
        # Add user message to chat context
        chat_context.messages.append(ChatMessage(role="user", content=text))
        
        # Get LLM response
        llm = openai.LLM(model="gpt-4")  # Changed to gpt-4 for better reliability
        logger.info("Getting LLM response...")
        response_stream = llm.chat(chat_ctx=chat_context)
        
        # Collect the full response
        full_response = []
        async for chunk in response_stream:
            logger.debug(f"Response chunk: {chunk}")
            if isinstance(chunk, str):
                full_response.append(chunk)
            elif hasattr(chunk, 'choices') and chunk.choices:
                # Extract content from ChatChunk
                delta = chunk.choices[0].delta
                if hasattr(delta, 'content') and delta.content is not None:
                    full_response.append(delta.content)
        
        # Join all response chunks
        final_response = ''.join(full_response)
        logger.info(f"LLM Response: {final_response}")
        
        if not final_response.strip():
            logger.error("Received empty response from LLM")
            return
            
        # Add assistant's response to chat context
        chat_context.messages.append(ChatMessage(role="assistant", content=final_response))
        
        # Send response as a chat message
        chat = rtc.ChatManager(room)
        await chat.send_message(final_response)
        
        # Create a speech event for the LLM response
        class Alternative:
            def __init__(self, text, confidence=1.0):
                self.text = text
                self.confidence = confidence

        class SpeechEventResponse:
            def __init__(self, text):
                self.type = stt.SpeechEventType.FINAL_TRANSCRIPT
                self.alternatives = [Alternative(f"Assistant: {text}")]
                self.recognition_usage = None

        # Forward the response using the same event structure
        response_event = SpeechEventResponse(final_response)
        stt_forwarder.update(response_event)
        
        logger.info(f"Response sent to chat and transcription: {final_response}")
    except Exception as e:
        logger.error(f"Error in _process_with_llm: {str(e)}", exc_info=True)

async def _detect_question_intent(text: str) -> bool:
    """Use LLM to detect if the text is a question or seeks information."""
    try:
        llm = openai.LLM()
        
        system_prompt = """You are a question and intent detector. Your task is to analyze text and determine if it:
1. Contains a question (explicit or implicit)
2. Seeks information or clarification
3. Requests assistance or help
4. Expresses curiosity or desire to learn

Respond with ONLY 'true' if any of these conditions are met, or 'false' if none are met."""
        
        context = ChatContext(messages=[
            ChatMessage(role="system", content=system_prompt),
            ChatMessage(role="user", content=text)
        ])
        
        # Get the response stream
        response_stream = llm.chat(chat_ctx=context)
        
        # Process the response
        response_text = ""
        async for chunk in response_stream:
            # Extract content from the ChatChunk
            if chunk.choices and chunk.choices[0].delta.content is not None:
                response_text += chunk.choices[0].delta.content
        
        result = response_text.strip().lower() == "true"
        logger.debug(f"Question detection result for '{text}': {result} (response: {response_text})")
        return result
        
    except Exception as e:
        logger.error(f"Error in question detection: {str(e)}", exc_info=True)
        # Fall back to simple detection in case of error
        return "?" in text or any(keyword in text.lower() for keyword in ["what", "how", "why", "can", "could", "would"])

async def _forward_transcription(
    stt_stream: stt.SpeechStream, stt_forwarder: transcription.STTSegmentsForwarder, room: rtc.Room
):
    """Forward the transcription to the client and process with LLM if needed"""
    async for ev in stt_stream:
        try:
            if ev.type == stt.SpeechEventType.INTERIM_TRANSCRIPT:
                pass
            elif ev.type == stt.SpeechEventType.FINAL_TRANSCRIPT:
                text = ev.alternatives[0].text
                print(" -> ", text)
                
                # First forward the transcription
                stt_forwarder.update(ev)
                
                # Use LLM-based question detection
                if await _detect_question_intent(text):
                    logger.info(f"Question/Intent detected: {text}")
                    await _process_with_llm(text, room, stt_forwarder)
                
            elif ev.type == stt.SpeechEventType.RECOGNITION_USAGE:
                logger.debug(f"metrics: {ev.recognition_usage}")
                stt_forwarder.update(ev)
        except Exception as e:
            logger.error(f"Error in _forward_transcription: {str(e)}", exc_info=True)

async def entrypoint(ctx: JobContext):
    logger.info(f"starting transcriber (speech to text) example, room: {ctx.room.name}")
    # this example uses OpenAI Whisper, but you can use assemblyai, deepgram, google, azure, etc.
    stt_impl = openai.STT()

    if not stt_impl.capabilities.streaming:
        # wrap with a stream adapter to use streaming semantics
        stt_impl = stt.StreamAdapter(
            stt=stt_impl,
            vad=silero.VAD.load(
                min_silence_duration=0.75,
                # silence_padding_duration=0.5,
            ),
        )

    async def transcribe_track(participant: rtc.RemoteParticipant, track: rtc.Track):
        audio_stream = rtc.AudioStream(track)
        stt_forwarder = transcription.STTSegmentsForwarder(
            room=ctx.room, participant=participant, track=track
        )

        stt_stream = stt_impl.stream()
        asyncio.create_task(_forward_transcription(stt_stream, stt_forwarder, ctx.room))

        async for ev in audio_stream:
            stt_stream.push_frame(ev.frame)

    @ctx.room.on("track_subscribed")
    def on_track_subscribed(
        track: rtc.Track,
        publication: rtc.TrackPublication,
        participant: rtc.RemoteParticipant,
    ):
        # spin up a task to transcribe each track
        if track.kind == rtc.TrackKind.KIND_AUDIO:
            asyncio.create_task(transcribe_track(participant, track))

    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))