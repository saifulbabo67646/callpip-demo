import asyncio
import logging
from typing import Annotated

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
from livekit.plugins import deepgram, openai, silero, elevenlabs

from screen_agent.dual_audio_agent import DualAudioAgent
from screen_agent.interview_agent_output import InterviewAgentOutput

load_dotenv(dotenv_path=".env.local")
logger = logging.getLogger("interview-assistant")


class InterviewAssistantFunction(llm.FunctionContext):
    """Functions that the interview assistant can call."""

    @llm.ai_callable(
        description="Analyze interview question and provide guidance"
    )
    async def analyze_question(
        self,
        question: Annotated[
            str,
            llm.TypeInfo(
                description="The interview question to analyze"
            )
        ],
    ) -> str:
        """Analyze an interview question and provide guidance on how to answer."""
        # This would typically involve more sophisticated analysis
        return f"This question is asking about your experience and skills. Focus on providing specific examples."

    @llm.ai_callable(
        description="Suggest technical references or documentation"
    )
    async def suggest_references(
        self,
        topic: Annotated[
            str,
            llm.TypeInfo(
                description="The technical topic to find references for"
            )
        ],
    ) -> str:
        """Suggest relevant technical documentation or references."""
        # This would typically involve a knowledge base lookup
        return f"For {topic}, you might want to reference the official documentation at docs.example.com"


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


async def entrypoint(ctx: JobContext):
    initial_ctx = llm.ChatContext().append(
        role="system",
        text=(
            "You are an AI interview assistant created to help job candidates during technical interviews. "
            "Your role is to listen to the interviewer's questions from screen share and provide helpful suggestions "
            "to the candidate. Keep your responses concise and practical. Focus on key technical concepts and "
            "best practices when providing guidance. Monitor both the interviewer's questions and the candidate's "
            "responses to provide contextual assistance."
        ),
    )

    logger.info(f"connecting to room {ctx.room.name}")
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)

    # Initialize variables for participants
    screen_participant = None
    mic_participant = None

    async def handle_participant(participant: rtc.RemoteParticipant):
        nonlocal screen_participant, mic_participant
        
        # Wait for tracks to be published
        for track in participant.tracks.values():
            if track.source == rtc.TrackSource.SCREEN_SHARE:
                screen_participant = participant
                logger.info(f"Screen share participant detected: {participant.identity}")
            elif track.source == rtc.TrackSource.MICROPHONE:
                mic_participant = participant
                logger.info(f"Microphone participant detected: {participant.identity}")

    # Set up participant connection handler
    ctx.room.on("participant_connected", handle_participant)
    
    # Handle existing participants
    for participant in ctx.room.participants.values():
        await handle_participant(participant)

    # Wait until we have both participants
    while not (screen_participant and mic_participant):
        await asyncio.sleep(1)
        logger.info("Waiting for both screen share and microphone participants...")

    logger.info(f"Starting interview assistant for screen: {screen_participant.identity} and mic: {mic_participant.identity}")

    stt_impl = deepgram.STT()
    tts_impl = elevenlabs.TTS(api_key="dummy_key")  # Replace with actual key in production

    agent = DualAudioAgent(
        vad=silero.VAD.load(),
        stt=stt_impl,
        llm=openai.LLM(model="gpt-4"),  # Using GPT-4 for better interview assistance
        tts=tts_impl,
        chat_ctx=initial_ctx,
        fnc_ctx=InterviewAssistantFunction(),
    )

    # Set up metrics collection using the correct metrics implementation
    @agent.on("stt_metrics")
    def on_stt_metrics(m: metrics.STTMetrics):
        logger.info(f"STT metrics - latency: {m.latency:.2f}s, confidence: {m.confidence:.2f}")

    @agent.on("llm_metrics")
    def on_llm_metrics(m: metrics.LLMMetrics):
        logger.info(f"LLM metrics - latency: {m.latency:.2f}s, tokens: {m.total_tokens}")

    # Start the agent with both audio streams
    await agent.start(
        room=ctx.room,
        screen_participant=screen_participant,
        mic_participant=mic_participant,
    )

    # Set up event handlers for transcripts
    @agent.on("screen_transcript")
    def handle_screen_transcript(transcript: str):
        logger.info(f"Interviewer: {transcript}")

    @agent.on("mic_transcript")
    def handle_mic_transcript(transcript: str):
        logger.info(f"Candidate: {transcript}")

    try:
        # Keep the agent running
        await ctx.wait_for_disconnect()
    finally:
        await agent.aclose()


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))