from livekit.agents.pipeline import VoicePipelineAgent
from livekit.agents.pipeline.human_input import HumanInput
from livekit.agents.pipeline.agent_output import AgentOutput
from livekit import rtc
from livekit.agents import stt, vad, utils, transcription
from livekit.agents import llm
import asyncio
import logging

logger = logging.getLogger("voice-assistant")

class CustomHumanInput(HumanInput):
    """Custom human input that handles screen share audio"""
    
    def _subscribe_to_microphone(self, *args, **kwargs) -> None:
        """
        Subscribe to the participant microphone and screen share audio if found.
        """
        for publication in self._participant.track_publications.values():
            # Accept both microphone and screen share audio
            if (
                publication.source == rtc.TrackSource.SOURCE_MICROPHONE or
                publication.source == rtc.TrackSource.SOURCE_SCREENSHARE_AUDIO
            ):
                if not publication.subscribed:
                    publication.set_subscribed(True)

                track: rtc.RemoteAudioTrack | None = publication.track  # type: ignore
                if track is not None and track != self._subscribed_track:
                    self._subscribed_track = track
                    if self._recognize_atask is not None:
                        self._recognize_atask.cancel()

                    self._recognize_atask = asyncio.create_task(
                        self._recognize_task(rtc.AudioStream(track, sample_rate=16000))
                    )
                
    @utils.log_exceptions(logger=logger)
    async def _recognize_task(self, audio_stream: rtc.AudioStream) -> None:
        """
        Receive the frames from the user audio stream and detect voice activity.
        """
        vad_stream = self._vad.stream()
        stt_stream = self._stt.stream()

        def _before_forward(
            fwd: transcription.STTSegmentsForwarder, transcription: rtc.Transcription
        ):
            if not self._transcription:
                transcription.segments = []
            return transcription

        stt_forwarder = transcription.STTSegmentsForwarder(
            room=self._room,
            participant=self._participant,
            track=self._subscribed_track,
            before_forward_cb=_before_forward,
        )

        async def _audio_stream_co() -> None:
            async for ev in audio_stream:
                stt_stream.push_frame(ev.frame)
                vad_stream.push_frame(ev.frame)

        async def _vad_stream_co() -> None:
            async for ev in vad_stream:
                if ev.type == vad.VADEventType.START_OF_SPEECH:
                    self._speaking = True
                    self.emit("start_of_speech", ev)
                elif ev.type == vad.VADEventType.INFERENCE_DONE:
                    self._speech_probability = ev.probability
                    self.emit("vad_inference_done", ev)
                elif ev.type == vad.VADEventType.END_OF_SPEECH:
                    self._speaking = False
                    self.emit("end_of_speech", ev)

        async def _stt_stream_co() -> None:
            async for ev in stt_stream:
                stt_forwarder.update(ev)

                if ev.type == stt.SpeechEventType.FINAL_TRANSCRIPT:
                    self.emit("final_transcript", ev)
                elif ev.type == stt.SpeechEventType.INTERIM_TRANSCRIPT:
                    self.emit("interim_transcript", ev)

        tasks = [
            asyncio.create_task(_audio_stream_co()),
            asyncio.create_task(_vad_stream_co()),
            asyncio.create_task(_stt_stream_co()),
        ]
        
        try:
            await asyncio.gather(*tasks)
        except asyncio.CancelledError:
            logger.info("Speech recognition task cancelled")
            raise
        except Exception as e:
            logger.error(f"Error in recognize task: {str(e)}", exc_info=True)
        finally:
            try:
                await stt_stream.aclose()
            except Exception as e:
                logger.error(f"Error closing STT stream: {str(e)}", exc_info=True)

class CustomVoicePipelineAgent(VoicePipelineAgent):
    """Custom voice pipeline agent that handles screen share audio"""
    
    def start(self, room: rtc.Room, participant: rtc.RemoteParticipant | str | None = None) -> None:
        """Override start to use CustomHumanInput"""
        if self._started:
            raise RuntimeError("voice assistant already started")

        room.on("participant_connected", self._on_participant_connected)
        self._room, self._participant = room, participant

        if participant is not None:
            if isinstance(participant, rtc.RemoteParticipant):
                self._link_participant(participant.identity)
            else:
                self._link_participant(participant)
        else:
            # no participant provided, try to find the first participant in the room
            for participant in self._room.remote_participants.values():
                self._link_participant(participant.identity)
                break

        self._started = True
        self._main_atask = asyncio.create_task(self._main_task())

    def _link_participant(self, identity: str) -> None:
        """Override _link_participant to use CustomHumanInput"""
        participant = self._room.remote_participants.get(identity)
        if participant is None:
            logger.error("_link_participant must be called with a valid identity")
            return

        self._human_input = CustomHumanInput(
            room=self._room,
            vad=self._vad,
            stt=self._stt,
            participant=participant,
            transcription=self._opts.transcription.user_transcription,
        )
        
        # Connect human input events to agent handlers
        self._human_input.on("start_of_speech", self._on_start_of_speech)
        self._human_input.on("vad_inference_done", self._on_vad_inference_done)
        self._human_input.on("end_of_speech", self._on_end_of_speech)
        self._human_input.on("interim_transcript", self._on_interim_transcript)
        self._human_input.on("final_transcript", self._on_final_transcript)
