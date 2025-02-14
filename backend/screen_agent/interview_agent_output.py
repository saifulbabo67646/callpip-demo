from __future__ import annotations

import asyncio
from typing import AsyncIterable, Optional

from livekit import rtc
from livekit.agents import tts, utils
from livekit.agents.llm import LLMStream

from .agent_output import AgentOutput, SynthesisHandle
from .log import logger

class InterviewAgentOutput(AgentOutput):
    """Specialized agent output for interview assistance scenarios."""
    
    def __init__(
        self,
        *,
        room: rtc.Room,
        tts: tts.TTS,
        transcription: bool = True,
        response_speed: float = 1.0,
    ) -> None:
        super().__init__(room=room, tts=tts, transcription=transcription)
        self._response_speed = response_speed
        self._current_suggestions: list[str] = []

    async def synthesize(
        self,
        source: str | LLMStream | AsyncIterable[str],
        *,
        speech_id: str,
    ) -> SynthesisHandle:
        """Synthesize agent response with interview-specific processing."""
        
        # Process the response for interview context
        if isinstance(source, str):
            processed_response = self._process_interview_response(source)
            source = processed_response
        elif isinstance(source, LLMStream):
            # Handle streaming responses
            source = self._stream_interview_response(source)
        
        return await super().synthesize(source, speech_id=speech_id)

    def _process_interview_response(self, text: str) -> str:
        """Process the agent's response for interview context."""
        # Add interview-specific markers and formatting
        # This could include prefixes like "Suggested Response:" or formatting for clarity
        return f"Suggested Response: {text}"

    async def _stream_interview_response(self, stream: LLMStream) -> AsyncIterable[str]:
        """Process streaming responses for interview context."""
        buffer = ""
        
        async for chunk in stream:
            buffer += chunk
            
            # Process complete sentences or thoughts
            if any(p in buffer for p in ".!?"):
                yield f"Suggestion: {buffer}"
                buffer = ""
                
        if buffer:  # Yield any remaining content
            yield f"Suggestion: {buffer}"

    def add_suggestion(self, suggestion: str) -> None:
        """Add a new suggestion to the current list."""
        self._current_suggestions.append(suggestion)
        
    def clear_suggestions(self) -> None:
        """Clear all current suggestions."""
        self._current_suggestions.clear()
        
    def get_current_suggestions(self) -> list[str]:
        """Get the current list of suggestions."""
        return self._current_suggestions.copy()
