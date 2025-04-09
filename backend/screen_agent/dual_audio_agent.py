from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Optional, Literal

from livekit import rtc
from livekit.agents import metrics, stt, vad
from livekit.agents.llm import LLM, ChatContext
from livekit.agents.types import AgentState

from .pipeline_agent import VoicePipelineAgent
from .human_input import HumanInput
from .agent_output import AgentOutput
from .log import logger

EventTypes = Literal[
    "screen_started_speaking",
    "screen_stopped_speaking",
    "mic_started_speaking",
    "mic_stopped_speaking",
    "screen_transcript",
    "mic_transcript",
    "agent_speaking",
    "agent_stopped",
]

@dataclass
class DualAudioTranscript:
    text: str
    is_final: bool
    source: Literal["screen", "mic"]
    timestamp: float

class DualAudioAgent(VoicePipelineAgent):
    """An agent that handles both screen share audio and microphone audio streams."""
    
    def __init__(
        self,
        *,
        vad: vad.VAD,
        stt: stt.STT,
        llm: LLM,
        tts: tts.TTS,
        chat_ctx: Optional[ChatContext] = None,
        **kwargs
    ) -> None:
        super().__init__(
            vad=vad,
            stt=stt,
            llm=llm,
            tts=tts,
            chat_ctx=chat_ctx,
            **kwargs
        )
        
        self._screen_input: Optional[HumanInput] = None
        self._mic_input: Optional[HumanInput] = None
        
        self._screen_transcript = ""
        self._mic_transcript = ""
        self._transcripts: list[DualAudioTranscript] = []

    async def start(
        self,
        room: rtc.Room,
        screen_participant: rtc.RemoteParticipant | str,
        mic_participant: rtc.RemoteParticipant | str,
    ) -> None:
        """Start the dual audio agent with both screen and mic participants."""
        
        self._started = True
        
        # Set up screen share audio input
        screen_participant = await self._get_participant(room, screen_participant)
        self._screen_input = HumanInput(
            room=room,
            vad=self._vad,
            stt=self._stt,
            participant=screen_participant,
            transcription=True,
        )
        
        # Set up microphone audio input
        mic_participant = await self._get_participant(room, mic_participant)
        self._mic_input = HumanInput(
            room=room,
            vad=self._vad,
            stt=self._stt,
            participant=mic_participant,
            transcription=True,
        )
        
        # Set up event handlers for screen audio
        self._screen_input.on("start_of_speech", lambda _: self._on_screen_speech_start())
        self._screen_input.on("end_of_speech", lambda _: self._on_screen_speech_end())
        self._screen_input.on("final_transcript", self._on_screen_final_transcript)
        self._screen_input.on("interim_transcript", self._on_screen_interim_transcript)
        
        # Set up event handlers for mic audio
        self._mic_input.on("start_of_speech", lambda _: self._on_mic_speech_start())
        self._mic_input.on("end_of_speech", lambda _: self._on_mic_speech_end())
        self._mic_input.on("final_transcript", self._on_mic_final_transcript)
        self._mic_input.on("interim_transcript", self._on_mic_interim_transcript)
        
        # Initialize agent output
        self._agent_output = AgentOutput(
            room=room,
            tts=self._tts,
            transcription=self._opts.transcription.agent_transcription,
        )
        
        self._update_state(AgentState.IDLE)
        self._main_task = asyncio.create_task(self._main_task())

    def _on_screen_speech_start(self) -> None:
        """Handle screen share speech start event."""
        self.emit("screen_started_speaking")
        self._update_state(AgentState.LISTENING)

    def _on_screen_speech_end(self) -> None:
        """Handle screen share speech end event."""
        self.emit("screen_stopped_speaking")
        if not self._mic_input or not self._mic_input._speaking:
            self._update_state(AgentState.PROCESSING)

    def _on_mic_speech_start(self) -> None:
        """Handle microphone speech start event."""
        self.emit("mic_started_speaking")

    def _on_mic_speech_end(self) -> None:
        """Handle microphone speech end event."""
        self.emit("mic_stopped_speaking")

    def _on_screen_final_transcript(self, transcript: str) -> None:
        """Handle screen share final transcript."""
        self._screen_transcript = transcript
        self._add_transcript(transcript, True, "screen")
        self.emit("screen_transcript", transcript)
        
        # Process screen audio for agent response
        self._chat_ctx.add_user_message(transcript)
        self._synthesize_agent_reply()

    def _on_screen_interim_transcript(self, transcript: str) -> None:
        """Handle screen share interim transcript."""
        self._add_transcript(transcript, False, "screen")
        self.emit("screen_transcript", transcript)

    def _on_mic_final_transcript(self, transcript: str) -> None:
        """Handle microphone final transcript."""
        self._mic_transcript = transcript
        self._add_transcript(transcript, True, "mic")
        self.emit("mic_transcript", transcript)

    def _on_mic_interim_transcript(self, transcript: str) -> None:
        """Handle microphone interim transcript."""
        self._add_transcript(transcript, False, "mic")
        self.emit("mic_transcript", transcript)

    def _add_transcript(self, text: str, is_final: bool, source: Literal["screen", "mic"]) -> None:
        """Add a new transcript to the history."""
        transcript = DualAudioTranscript(
            text=text,
            is_final=is_final,
            source=source,
            timestamp=asyncio.get_event_loop().time()
        )
        self._transcripts.append(transcript)

    async def aclose(self) -> None:
        """Close the dual audio agent."""
        if self._closed:
            return
            
        self._closed = True
        
        if self._screen_input:
            await self._screen_input.aclose()
        if self._mic_input:
            await self._mic_input.aclose()
            
        await super().aclose()

    @staticmethod
    async def _get_participant(
        room: rtc.Room, participant: rtc.RemoteParticipant | str
    ) -> rtc.RemoteParticipant:
        """Get a participant from the room."""
        if isinstance(participant, str):
            participant_obj = room.remote_participants.get(participant)
            if not participant_obj:
                raise ValueError(f"Participant {participant} not found in room")
            return participant_obj
        return participant
