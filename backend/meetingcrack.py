import asyncio
import logging
from typing import Optional
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
    LLM,
)
from livekit.plugins import openai, silero
from livekit.agents.transcription import STTSegmentsForwarder

# Load environment variables
load_dotenv(dotenv_path=".env.local")

# Configure logging
logger = logging.getLogger("meetingcrack")

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import json

class QuestionType(Enum):
    TECHNICAL = "technical"
    BEHAVIORAL = "behavioral"
    CLARIFICATION = "clarification"
    EXPERIENCE = "experience"
    SYSTEM_DESIGN = "system_design"
    GENERAL = "general"

@dataclass
class InterviewContext:
    """Maintains the context of the interview"""
    question_type: QuestionType
    technologies: List[str]
    complexity: str
    previous_context: str
    requirements: List[str]

@dataclass
class AnswerFormat:
    """Defines how the answer should be structured"""
    format_type: str  # code, explanation, steps, example, etc.
    language: Optional[str] = None  # for code responses
    include_examples: bool = False
    include_references: bool = False

class LLMFunctionRegistry:
    """Registry for LLM functions that can be called during response generation"""
    
    @staticmethod
    def get_interview_functions() -> List[Dict[str, Any]]:
        """Get the list of available functions for interview responses"""
        return [
            {
                "name": "answer_technical_question",
                "description": "Provide a technical answer with optional code examples",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "question_type": {
                            "type": "string",
                            "enum": ["algorithm", "data_structure", "system_design", "coding", "concept"],
                        },
                        "complexity": {
                            "type": "string",
                            "enum": ["basic", "intermediate", "advanced"],
                        },
                        "include_code": {"type": "boolean"},
                        "programming_language": {"type": "string"},
                        "explanation": {"type": "string"},
                        "code_example": {"type": "string"},
                        "time_complexity": {"type": "string"},
                        "space_complexity": {"type": "string"},
                    },
                    "required": ["question_type", "complexity", "explanation"],
                },
            },
            {
                "name": "answer_behavioral_question",
                "description": "Provide a structured behavioral answer using the STAR method",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "situation": {"type": "string"},
                        "task": {"type": "string"},
                        "action": {"type": "string"},
                        "result": {"type": "string"},
                        "learning": {"type": "string"},
                    },
                    "required": ["situation", "task", "action", "result"],
                },
            },
            {
                "name": "provide_experience_example",
                "description": "Share relevant experience with specific technologies or concepts",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "technology": {"type": "string"},
                        "experience_level": {"type": "string"},
                        "project_example": {"type": "string"},
                        "challenges": {"type": "string"},
                        "solutions": {"type": "string"},
                    },
                    "required": ["technology", "experience_level", "project_example"],
                },
            },
        ]

class ResponseFormatter:
    """Formats the LLM response based on the question type and context"""
    
    @staticmethod
    def format_technical_response(response: Dict[str, Any]) -> str:
        formatted = []
        formatted.append(f"Technical Response:\n")
        formatted.append(f"{response['explanation']}\n")
        
        if response.get('code_example'):
            formatted.append(f"\nCode Example:\n```{response.get('programming_language', '')}")
            formatted.append(response['code_example'])
            formatted.append("```\n")
            
        if response.get('time_complexity'):
            formatted.append(f"\nTime Complexity: {response['time_complexity']}")
        if response.get('space_complexity'):
            formatted.append(f"Space Complexity: {response['space_complexity']}")
            
        return "\n".join(formatted)
    
    @staticmethod
    def format_behavioral_response(response: Dict[str, Any]) -> str:
        formatted = []
        formatted.append("Let me share a relevant experience:\n")
        formatted.append(f"\nSituation:\n{response['situation']}")
        formatted.append(f"\nTask:\n{response['task']}")
        formatted.append(f"\nAction:\n{response['action']}")
        formatted.append(f"\nResult:\n{response['result']}")
        if response.get('learning'):
            formatted.append(f"\nKey Learning:\n{response['learning']}")
        return "\n".join(formatted)
    
    @staticmethod
    def format_experience_response(response: Dict[str, Any]) -> str:
        formatted = []
        formatted.append(f"Experience with {response['technology']} ({response['experience_level']}):\n")
        formatted.append(f"\nProject Example:\n{response['project_example']}")
        if response.get('challenges'):
            formatted.append(f"\nChallenges Faced:\n{response['challenges']}")
        if response.get('solutions'):
            formatted.append(f"\nSolutions Implemented:\n{response['solutions']}")
        return "\n".join(formatted)

class QuestionDetector:
    """Sophisticated question detector using multiple heuristics and patterns"""
    
    def __init__(self):
        # Direct question patterns (highest weight)
        self.direct_patterns = [
            QuestionPattern(r"\?", 1.0, is_regex=True),
            QuestionPattern(r"^(what|how|why|when|where|who|which|whose|whom)\b", 0.9, is_regex=True),
            QuestionPattern(r"^(can|could|would|will|should|do|does|did|is|are|was|were)\b", 0.85, is_regex=True),
        ]
        
        # Indirect question patterns (medium weight)
        self.indirect_patterns = [
            QuestionPattern(r"\b(tell me|explain|describe|elaborate|clarify)\b", 0.7, is_regex=True),
            QuestionPattern(r"\b(i('d| would) like to know|i wonder|i('m| am) curious)\b", 0.65, is_regex=True),
            QuestionPattern(r"\b(share|provide|give me|help me understand)\b", 0.6, is_regex=True),
        ]
        
        # Context-dependent patterns (lower weight)
        self.contextual_patterns = [
            QuestionPattern(r"\b(right|correct|isn't it|aren't they)\b", 0.4, is_regex=True),
            QuestionPattern(r"\b(for example|such as|like what)\b", 0.35, is_regex=True),
        ]
        
        # Technical interview specific patterns
        self.technical_patterns = [
            QuestionPattern(r"\b(implement|design|code|program|write|solve)\b", 0.75, is_regex=True),
            QuestionPattern(r"\b(what happens|what would happen|what's the output)\b", 0.8, is_regex=True),
            QuestionPattern(r"\b(time complexity|space complexity|performance|optimize)\b", 0.7, is_regex=True),
            QuestionPattern(r"\b(experience with|familiar with|worked with)\b", 0.65, is_regex=True),
        ]
        
        # Behavioral interview specific patterns
        self.behavioral_patterns = [
            QuestionPattern(r"\b(tell me about a time|describe a situation|give an example)\b", 0.85, is_regex=True),
            QuestionPattern(r"\b(how did you|what did you|how would you|what would you)\b", 0.8, is_regex=True),
            QuestionPattern(r"\b(challenge|conflict|difficult|success|fail|learn|handle|manage)\b", 0.6, is_regex=True),
        ]
        
        # Follow-up indicators
        self.followup_patterns = [
            QuestionPattern(r"\b(and then|what about|how about|what if)\b", 0.5, is_regex=True),
            QuestionPattern(r"^(okay|so|and|but)\b", 0.3, is_regex=True),
        ]

    def is_question(self, text: str, context: Optional[List[str]] = None) -> Tuple[bool, float, Set[str]]:
        """
        Determine if the text is a question using multiple heuristics.
        
        Args:
            text: The text to analyze
            context: Optional list of previous conversation turns for context
            
        Returns:
            Tuple of (is_question: bool, confidence: float, matched_patterns: Set[str])
        """
        text = text.strip()
        if not text:
            return False, 0.0, set()
            
        confidence = 0.0
        matched_patterns = set()
        
        # Check all pattern types
        pattern_groups = [
            (self.direct_patterns, "direct"),
            (self.indirect_patterns, "indirect"),
            (self.contextual_patterns, "contextual"),
            (self.technical_patterns, "technical"),
            (self.behavioral_patterns, "behavioral"),
            (self.followup_patterns, "followup"),
        ]
        
        for patterns, group_name in pattern_groups:
            for pattern in patterns:
                if pattern.matches(text):
                    confidence = max(confidence, pattern.weight)
                    matched_patterns.add(f"{group_name}:{pattern.pattern}")
        
        # Context-based adjustments
        if context:
            prev_turn = context[-1].lower() if context else ""
            
            # Boost confidence if this appears to be a follow-up to a previous question
            if any(p.matches(prev_turn) for p in self.direct_patterns):
                confidence = min(1.0, confidence + 0.1)
                matched_patterns.add("context:follow_up_to_question")
            
            # Boost confidence if this is a clarifying statement after an answer
            if any(word in prev_turn for word in ["because", "therefore", "so", "thus"]):
                confidence = min(1.0, confidence + 0.05)
                matched_patterns.add("context:clarification")
        
        # Additional heuristics
        if len(text.split()) <= 4 and not any(p.matches(text) for p in self.direct_patterns):
            confidence *= 0.8  # Reduce confidence for very short non-direct questions
            matched_patterns.add("heuristic:short_text")
            
        if text.isupper():
            confidence *= 0.9  # Slightly reduce confidence for all-caps text
            matched_patterns.add("heuristic:all_caps")
        
        # Log the detection details for debugging
        logger.debug(
            f"Question detection: confidence={confidence:.2f}, patterns={matched_patterns}",
            extra={"text": text, "is_question": confidence >= 0.4}
        )
        
        return confidence >= 0.4, confidence, matched_patterns

@dataclass
class QuestionPattern:
    """Represents a pattern that indicates a question or information request"""
    pattern: str
    weight: float  # How strongly this pattern indicates a question (0.0 to 1.0)
    is_regex: bool = False
    
    def matches(self, text: str) -> bool:
        if self.is_regex:
            return bool(re.search(self.pattern, text, re.IGNORECASE))
        return self.pattern.lower() in text.lower()

class MeetingAssistant:
    """Assistant that helps with interview questions by transcribing and providing AI responses"""
    
    def __init__(self):
        self.chat_context = ChatContext(
            messages=[
                ChatMessage(
                    role="system",
                    content=(
                        "You are an AI interview assistant. Your role is to:\n"
                        "1. Listen to interview questions and provide helpful suggestions for answers\n"
                        "2. Keep track of the conversation context\n"
                        "3. Provide concise, professional responses\n"
                        "4. Consider both technical accuracy and communication style\n"
                        "5. Adapt responses based on the interviewer's reactions\n"
                        "Use the provided functions to structure your responses appropriately."
                    ),
                )
            ]
        )
        
        # Initialize LLM with function calling
        self.llm = openai.LLM(
            model="gpt-4",
            functions=LLMFunctionRegistry.get_interview_functions()
        )
        
        # Initialize other components
        self.stt_impl = openai.STT()
        self.vad = silero.VAD.load(
            min_silence_duration=1.5,
        )
        self.stt = stt.StreamAdapter(
            stt=self.stt_impl,
            vad=self.vad,
        )
        
        self.question_detector = QuestionDetector()
        self.conversation_history: List[str] = []
        self.current_interview_context: Optional[InterviewContext] = None

    async def process_with_llm(self, text: str, room: rtc.Room, stt_forwarder: transcription.STTSegmentsForwarder):
        """Process transcribed text with LLM and generate structured response"""
        try:
            logger.info(f"Processing text with LLM: {text}")
            
            # Add the interviewer's question to chat context
            self.chat_context.messages.append(
                ChatMessage(role="user", content=f"Interviewer: {text}")
            )
            
            # Get LLM response with function calling
            response_stream = self.llm.chat(
                chat_ctx=self.chat_context,
                temperature=0.7,
                max_tokens=1000
            )
            
            # Process the response and handle function calls
            full_response = ""
            async for chunk in response_stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    full_response += chunk.choices[0].delta.content
                elif chunk.choices and chunk.choices[0].delta.function_call:
                    # Handle function call
                    function_call = chunk.choices[0].delta.function_call
                    formatted_response = await self._handle_function_call(function_call)
                    full_response += formatted_response
            
            # Add AI response to chat context
            self.chat_context.messages.append(
                ChatMessage(role="assistant", content=full_response)
            )
            
            # Create a transcription event for the AI response
            response_event = transcription.TranscriptionEvent(
                type=transcription.TranscriptionEventType.FINAL,
                text=f"AI Assistant: {full_response}",
                source="assistant"
            )
            
            # Forward the response to the client
            stt_forwarder.update(response_event)
            logger.info(f"Response sent: {full_response}")
            
        except Exception as e:
            logger.error(f"Error in process_with_llm: {str(e)}", exc_info=True)

    async def _handle_function_call(self, function_call: Dict[str, Any]) -> str:
        """Handle LLM function calls and format the response"""
        try:
            function_name = function_call["name"]
            arguments = json.loads(function_call["arguments"])
            
            if function_name == "answer_technical_question":
                return ResponseFormatter.format_technical_response(arguments)
            elif function_name == "answer_behavioral_question":
                return ResponseFormatter.format_behavioral_response(arguments)
            elif function_name == "provide_experience_example":
                return ResponseFormatter.format_experience_response(arguments)
            else:
                logger.warning(f"Unknown function call: {function_name}")
                return str(arguments)  # Fallback to raw response
                
        except Exception as e:
            logger.error(f"Error handling function call: {str(e)}", exc_info=True)
            return "I apologize, but I encountered an error processing the response. Could you please rephrase your question?"

    async def handle_transcription(
        self,
        stt_stream: stt.SpeechStream,
        stt_forwarder: transcription.STTSegmentsForwarder,
        room: rtc.Room
    ):
        """Handle transcription stream and process with LLM when appropriate"""
        
        async for ev in stt_stream:
            try:
                if ev.type == stt.SpeechEventType.FINAL_TRANSCRIPT:
                    text = ev.alternatives[0].text
                    logger.info(f"Transcribed: {text}")
                    
                    # Forward the transcription to the client
                    stt_forwarder.update(ev)
                    
                    # Add to conversation history
                    self.conversation_history.append(text)
                    if len(self.conversation_history) > 10:  # Keep last 10 turns
                        self.conversation_history.pop(0)
                    
                    # Process with LLM if it appears to be a question
                    is_question, confidence, patterns = self.question_detector.is_question(
                        text, 
                        context=self.conversation_history[:-1]
                    )
                    
                    if is_question:
                        logger.info(
                            f"Question detected (confidence: {confidence:.2f})",
                            extra={"patterns": patterns}
                        )
                        await self.process_with_llm(text, room, stt_forwarder)
                    
                elif ev.type == stt.SpeechEventType.RECOGNITION_USAGE:
                    logger.debug(f"Recognition metrics: {ev.recognition_usage}")
                    stt_forwarder.update(ev)
                    
            except Exception as e:
                logger.error(f"Error in handle_transcription: {str(e)}", exc_info=True)

async def entrypoint(ctx: JobContext):
    """Main entry point for the meeting assistant"""
    
    room = ctx.room
    assistant = MeetingAssistant()
    
    # Wait for the room to be ready
    await ctx.connect()
    logger.info(f"Connected to room: {room.name}")
    
    # Subscribe to all tracks automatically
    @room.on("track_subscribed")
    async def on_track_subscribed(track: rtc.Track, publication: rtc.TrackPublication, participant: rtc.RemoteParticipant):
        if track.kind == rtc.TrackKind.AUDIO:
            logger.info(f"Processing audio from participant: {participant.identity}")
            
            # Create a transcription forwarder
            stt_forwarder = transcription.STTSegmentsForwarder(
                room=room,
                participant_identity=participant.identity
            )
            
            # Start processing the audio
            stt_stream = assistant.stt.stream(track)
            await assistant.handle_transcription(stt_stream, stt_forwarder, room)
    
    try:
        # Keep the connection alive
        while True:
            await asyncio.sleep(1)
    except Exception as e:
        logger.error(f"Error in entrypoint: {str(e)}", exc_info=True)
    finally:
        await ctx.disconnect()

if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
