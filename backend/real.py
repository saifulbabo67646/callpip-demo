import asyncio
import logging
import json
import time
from typing import Annotated, Any, AsyncIterable

from dotenv import load_dotenv
from livekit import rtc
from livekit.agents import (
    AutoSubscribe,
    JobContext,
    JobProcess,
    WorkerOptions,
    cli,
    llm,
    metrics,
    stt,
    transcription,
)
# from livekit.agents.pipeline import VoicePipelineAgent
from livekit.plugins import deepgram, openai, silero, elevenlabs

from screen_agent import VoicePipelineAgent

load_dotenv(dotenv_path=".env.local")
logger = logging.getLogger("voice-assistant")


class AssistantFunction(llm.FunctionContext):
    """This class defines functions that the assistant can call."""

    @llm.ai_callable(
        description="Get the current weather for a specific location"
    )
    async def get_weather(
        self,
        location: Annotated[
            str,
            llm.TypeInfo(
                description="The location to get weather for"
            )
        ],
    ) -> str:
        """Get the weather for a specific location"""
        # This is a mock implementation
        return f"The weather in {location} is sunny and 22°C"

    @llm.ai_callable(
        description="Set a reminder for a specific time"
    )
    async def set_reminder(
        self,
        time: Annotated[
            str,
            llm.TypeInfo(
                description="The time for the reminder (e.g. '3pm tomorrow')"
            )
        ],
        message: Annotated[
            str,
            llm.TypeInfo(
                description="The reminder message"
            )
        ],
    ) -> str:
        """Set a reminder for a specific time"""
        # This is a mock implementation
        return f"Reminder set for {time}: {message}"


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


async def entrypoint(ctx: JobContext):
    initial_ctx = llm.ChatContext().append(
        role="system",
        text=(
            "You are a software engineer candidate in a job interview. Your responses should be: "
            "1. Professional and thoughtful, demonstrating your expertise while remaining humble "
            "2. Contextual - always listen carefully to the interviewer's questions and context before responding "
            "3. Clear and concise, avoiding technical jargon unless specifically relevant "
            "4. Based on the actual question asked - don't make assumptions or jump to conclusions "
            "5. Honest - if you need clarification or don't fully understand something, ask for clarification "
            "Remember to: "
            "- Wait for the complete question before responding "
            "- Consider the context of previous questions in the interview "
            "- Structure your responses clearly but naturally, as you would in a real conversation "
            "- Keep responses focused and relevant to what was actually asked"
        ),
    )

    logger.info(f"connecting to room {ctx.room.name}")
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)

    # wait for the first participant to connect
    participant = await ctx.wait_for_participant()
    logger.info(f"starting voice assistant for participant {participant.identity}")

    stt_impl = deepgram.STT()

    # Initialize VAD with interview-optimized parameters
    vad = silero.VAD.load(
        # Longer minimum speech duration since interview questions tend to be longer
        min_speech_duration=0.3,  # Increased from 0.05 to better handle natural speech
        # Longer silence duration to not cut off mid-sentence pauses
        min_silence_duration=0.8,  # Increased from 0.55 to allow for natural pauses
        # More padding to capture context
        prefix_padding_duration=0.8,  # Increased from 0.5 to catch early words
        # Longer buffer for longer questions
        max_buffered_speech=120.0,  # Doubled from 60.0 to handle longer speech segments
        # Slightly lower threshold to catch softer speech
        activation_threshold=0.45,  # Decreased from 0.5 to be more sensitive
        sample_rate=16000
    )

    # Intercept text before it goes to TTS
    def before_tts(agent: VoicePipelineAgent, text: str | AsyncIterable[str]) -> str | AsyncIterable[str]:
        if isinstance(text, str):
            logger.info(f"LLM Response (Single): {text}")
            # Send single response through data channel
            data = json.dumps({
                "type": "llm_response",
                "text": text,
                "timestamp": int(time.time() * 1000)
            }).encode()
            
            asyncio.create_task(
                agent._room.local_participant.publish_data(
                    payload=data,
                    reliable=True,
                    topic="llm_stream"
                )
            )
            return text
        else:
            async def log_stream():
                async for chunk in text:
                    logger.info(f"LLM Response (Stream): {chunk}")
                    # Send each chunk through data channel
                    data = json.dumps({
                        "type": "llm_response",
                        "text": chunk,
                        "timestamp": int(time.time() * 1000)
                    }).encode()
                    
                    await agent._room.local_participant.publish_data(
                        payload=data,
                        reliable=True,
                        topic="llm_stream"
                    )
                    yield chunk
            return log_stream()

    agent = VoicePipelineAgent(
        vad=vad,
        stt=stt_impl,
        llm=openai.LLM(model="gpt-4o-mini"),
        tts=elevenlabs.TTS(api_key="dummy_key"),  # Dummy key since we don't need TTS
        chat_ctx=initial_ctx,
        fnc_ctx=AssistantFunction(),
        before_tts_cb=before_tts,  # Add the TTS interceptor
        allow_interruptions=False,
    )

    # Create a handler to capture LLM stream before TTS
    @agent.on("user_speech_committed")
    def on_user_speech(message: llm.ChatMessage):
        logger.info(f"Processing complete utterance with LLM: {message.content}")
        # Create task to handle the LLM response
        # asyncio.create_task(get_llm_response(message.content))

    async def get_llm_response(user_text: str):
        # Get LLM response using agent's LLM and chat context
        logger.info(f"Getting LLM response for: {user_text}")
        
        # Add user message to chat context
        # agent.chat_ctx.append(role="user", text=user_text)
        
        # Get LLM response
        stream = agent.llm.chat(chat_ctx=agent.chat_ctx, fnc_ctx=agent.fnc_ctx)
        response_buffer = []
        
        async for chunk in stream:
            if not chunk.choices:
                continue
            choice = chunk.choices[0]
            
            # Handle content
            if choice.delta.content:
                response_buffer.append(choice.delta.content)
                logger.info(f"LLM stream chunk: {choice.delta.content}")
                
        complete_response = "".join(response_buffer)
        if complete_response:
            logger.info(f"Complete LLM response: {complete_response}")
            # Add assistant response to chat context
            # agent.chat_ctx.append(role="assistant", text=complete_response)

    # Handle function calls events
    @agent.on("function_calls_collected")
    def on_function_calls(function_calls):
        logger.info(f"Function calls collected: {function_calls}")

    @agent.on("function_calls_finished")
    def on_function_calls_finished(called_functions):
        for func in called_functions:
            logger.info(f"Function  finished with result: {func.result}")

    agent.start(ctx.room, participant)

    usage_collector = metrics.UsageCollector()

    @agent.on("metrics_collected")
    def _on_metrics_collected(mtrcs: metrics.AgentMetrics):
        metrics.log_metrics(mtrcs)
        usage_collector.collect(mtrcs)

    async def log_usage():
        summary = usage_collector.get_summary()
        logger.info(f"Usage: ${summary}")

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
                    
                    # Only forward the transcription, let VoicePipelineAgent handle LLM
                    stt_forwarder.update(ev)
                    
                elif ev.type == stt.SpeechEventType.END_OF_SPEECH:
                    logger.info("End of speech detected")
                    
                elif ev.type == stt.SpeechEventType.RECOGNITION_USAGE:
                    logger.debug(f"metrics: {ev.recognition_usage}")
                    stt_forwarder.update(ev)
            except Exception as e:
                logger.error(f"Error in _forward_transcription: {str(e)}", exc_info=True)

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
        
        # for publication in participant.track_publications.values():
        #     if publication.source == rtc.TrackSource.SOURCE_SCREENSHARE_AUDIO:
        #         continue
        #     #     logger.info(f"pulicationsssssssssss screen share: {publication}")
        #     #     asyncio.create_task(transcribe_track(participant, track))
        #     if publication.source != rtc.TrackSource.SOURCE_MICROPHONE:
        #         continue

        #     logger.info(f"pulicationsssssssssss screen share: {publication}")
        #     asyncio.create_task(transcribe_track(participant, track))

        # spin up a task to transcribe each track
        if track.kind == rtc.TrackKind.KIND_AUDIO and publication.source == rtc.TrackSource.SOURCE_MICROPHONE:
            asyncio.create_task(transcribe_track(participant, track))


    ctx.add_shutdown_callback(log_usage)


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))