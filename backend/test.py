import asyncio
import os
import logging
from typing import Annotated
# from dotenv import load_dotenv
from dotenv import load_dotenv
from livekit.plugins import deepgram, elevenlabs, openai, silero
from livekit import rtc
from livekit.agents import JobContext, WorkerOptions, cli
from livekit.agents.pipeline import VoicePipelineAgent
from livekit.agents.llm import ChatContext, ChatMessage
from livekit.agents import llm

# Configure logging
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

load_dotenv(dotenv_path=".env.local")

logger = logging.getLogger("chat-agent")


class Actions(llm.FunctionContext):
    """Actions"""

    def __init__(self):
        super().__init__()

    @llm.ai_callable()
    async def get_flight_info(self,
        flight_number: Annotated[str, llm.TypeInfo(description="Airline flight number")]):
        """
        Get flight information for a given flight number

        Args:
            flight_number: Airline flight number

        Returns:
            Flight information
        """
        return {"flight_number": flight_number, "departure": "LAX", "arrival": "JFK", "date": "2024-01-01", "time": "10:00 AM", "status": "On time"}

    @llm.ai_callable()
    async def get_booking_info(self, booking_number: Annotated[str, llm.TypeInfo(description="Airline booking number")]):
        """
        Get booking information for a given booking number

        Args:
            booking_number: Airline booking number

        Returns:
            Booking information
        """
        return {"booking_number": booking_number, 
                "flight_number": "AE" + booking_number, 
                "passenger_name": "John Doe",
                "passenger_email": "john.doe@example.com",
                "passenger_phone": "1234567890",
                "passenger_address": "123 Main St, Anytown, USA",
                "passenger_dob": "1990-01-01",
                "passenger_gender": "Male"
                }


class ChatAgent:
    def __init__(self):
        self.agent : VoicePipelineAgent | None = None

    @staticmethod
    def get_system_prompt() -> str:
        return """
        You are Alice, a helpful customer airlines service agent. Do not handle any other topics.

        # AVAILABLE FUNCTIONS
        You have access to the following functions to retrieve information when needed:
        - get_flight_info(flight_number: str): Returns flight information (departure, arrival, date, time, status)
        - get_booking_info(booking_number: str): Returns booking information (departure, arrival, date, time, status)
        """

    def initialize_agent(self):
        
        system_prompt = ChatAgent.get_system_prompt()
        logger.info(f"System prompt: {system_prompt}")
        
        initial_ctx = ChatContext()
        initial_ctx.append(role="system", text=system_prompt)

        call_actions = Actions()

        # Initialize agent without TTS since we don't need voice response
        self.agent = VoicePipelineAgent(
            vad=silero.VAD.load(),
            stt=deepgram.STT(model="nova-2-phonecall"),
            llm=openai.LLM(model="gpt-4o-mini"),
            tts=elevenlabs.TTS(api_key="asdfsdfasdfasdfasdfsadfasdf"),
            chat_ctx=initial_ctx,
            fnc_ctx=call_actions
        )

        # Set up audio event handlers
        # @self.agent.on("speech_start")
        # def on_speech_start():
        #     logger.info("Speech started")

        # @self.agent.on("speech_end")
        # def on_speech_end():
        #     logger.info("Speech ended")

        # @self.agent.on("transcription")
        # async def on_transcription(text: str):
        #     logger.info(f"Transcribed text: {text}")
        #     # Get response from LLM
        #     response = await self.chat_response(text)
        #     # Here you can emit the response to your frontend
        #     # For example, through a websocket or other communication channel
        #     logger.info(f"LLM Response: {response}")
        #     # Note: We don't call agent.say() anymore since we don't want voice response

    async def start_audio_session(self, room):
        """Start the agent with audio capabilities in the given room"""
        if self.agent is None:
            raise ValueError("Agent is not initialized")
        
        # Start the agent with audio processing
        self.agent.start(room)
        
        # Keep the connection alive
        while room.connection_state == rtc.ConnectionState.CONN_CONNECTED:
            await asyncio.sleep(1)

    async def chat_response(self, message: str) -> str:
        if self.agent is None:
            raise ValueError("Agent is not initialized")

        self.agent.chat_ctx.messages.append(ChatMessage.create(text=message, role="user"))  
        while True:
            # Get the stream and process chunks
            stream = self.agent.llm.chat(chat_ctx=self.agent.chat_ctx, fnc_ctx=self.agent.fnc_ctx)

            # Process response stream
            response_buffer = []
            tool_calls = []
            async for chunk in stream:
                if not chunk.choices:
                    continue
                choice = chunk.choices[0]
                
                if choice.delta.content:
                    response_buffer.append(choice.delta.content)
                if choice.delta.tool_calls:
                    has_function_calls = True
                    for fnc in choice.delta.tool_calls:
                        tool_calls.append(fnc)

            complete_response = "".join(response_buffer)
            if complete_response:
                self.agent.chat_ctx.messages.append(ChatMessage.create(text=complete_response, role="assistant"))
                return complete_response

            called_fncs = []
            for fnc in tool_calls:
                logger.debug(f"Function call: {fnc}")

                called_fnc = fnc.execute()
                called_fncs.append(called_fnc)
                logger.debug(f"Executing ai function: {fnc.function_info.name}")
                try:
                    # Add 10 second timeout for function execution
                    await asyncio.wait_for(called_fnc.task, timeout=2.0)
                except asyncio.TimeoutError:
                    logger.error(f"Function {fnc.function_info.name} timed out after 2 seconds")
                except Exception as e:
                    logger.error(f"Error executing ai function: {fnc.function_info.name} - {str(e)}")

            
            tool_calls_info = []
            tool_calls_results = []

            for called_fnc in called_fncs:
                # ignore the function calls that returns None
                if called_fnc.result is None and called_fnc.exception is None:
                    continue

                logger.debug(f"Called result function: {called_fnc}")
                tool_calls_info.append(called_fnc.call_info)
                tool_calls_results.append(
                    ChatMessage.create_tool_from_called_function(called_fnc)
                )

            if tool_calls_info:
                # create a nested speech handle
                extra_tools_messages = [
                    ChatMessage.create_tool_calls(tool_calls_info)
                ]
                extra_tools_messages.extend(tool_calls_results)

                # synthesize the tool speech with the chat ctx from llm_stream
                self.agent.chat_ctx.messages.extend(extra_tools_messages)
                # agent.chat_ctx.messages.extend(call_ctx.extra_chat_messages)
                logger.debug(self.agent.chat_ctx.messages)

                continue
            else:
                break

        logger.error("Agent unable to respond")
        return ""
    
    def reset(self):
        if self.agent is None:
            return
        self.agent.chat_ctx.messages = []
        self.agent = None

    def get_agent(self) -> VoicePipelineAgent:
        if self.agent is None:
            raise ValueError("Agent is not initialized")
        return self.agent
    
    def get_chat_ctx(self) -> ChatContext:
        if self.agent is None:
            raise ValueError("Agent is not initialized")
        return self.agent.chat_ctx
    
    def get_chat_messages(self) -> list[ChatMessage]:
        if self.agent is None:
            raise ValueError("Agent is not initialized")
        return self.agent.chat_ctx.messages


async def entrypoint(ctx: JobContext):
    """Main entrypoint for the agent"""
    room_name = ctx.room.name
    logger.info(f"Starting agent in room: {room_name}")

    # Connect to the room
    await ctx.connect()
    logger.info(f"Connected to room: {room_name}")

    # Initialize and start the chat agent
    chat_agent = ChatAgent()
    chat_agent.initialize_agent()
    await chat_agent.start_audio_session(ctx.room)

    @ctx.room.on("track_subscribed")
    def on_track_subscribed(
        track: rtc.Track,
        publication: rtc.TrackPublication,
        participant: rtc.RemoteParticipant,
    ):
        # spin up a task to transcribe each track
        if track.kind == rtc.TrackKind.KIND_AUDIO:
            # asyncio.create_task(transcribe_track(participant, track))
            pass


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))