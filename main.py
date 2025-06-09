# Script that uses a fixed interval window to determine mood, then pipes the result to OBS
import pyaudio
import numpy as np
import asyncio
import simpleobsws
import os

from typing import Self
from enum import Enum
from funasr import AutoModel


class Emotion(Enum):
    """
    Emotions mapping
    """

    ANGRY = 0
    DISGUSTED = 1
    FEARFUL = 2
    HAPPY = 3
    NEUTRAL = 4
    OTHER = 5
    SAD = 6
    SURPRISED = 7
    UNKNOWN = 8

    def from_index(number: int) -> Self:
        return [
            Emotion.ANGRY,
            Emotion.DISGUSTED,
            Emotion.FEARFUL,
            Emotion.HAPPY,
            Emotion.NEUTRAL,
            Emotion.OTHER,
            Emotion.SAD,
            Emotion.SURPRISED,
            Emotion.UNKNOWN,
        ][number]


# Scene mapping activation
SCENE_MAPPING = {
    Emotion.ANGRY: "VanorMad",
    Emotion.DISGUSTED: "VanorMad",
    Emotion.FEARFUL: "VanorD",
    Emotion.HAPPY: "Vanor",
    Emotion.NEUTRAL: "Vanor",
    Emotion.OTHER: "Vanor",
    Emotion.SAD: "VanorD",
    Emotion.SURPRISED: "VanorD",
    Emotion.UNKNOWN: "Vanor",
}

# All managed scenes
MANAGED_SCENES = ["VanorMad", "VanorD", "Vanor"]

# Recording options, don't touch this
FORMAT = pyaudio.paFloat32
CHANNELS = 1
RATE = 16_000
CHUNK = 1024
RECORD_SECONDS = 1
WINDOW_SECONDS = 5
SILENCE_THRESHOLD = 0.2

MODEL = AutoModel(
    model="emotion2vec/emotion2vec_plus_large",
    disable_update=True,
    # device="cuda:0", GPU is usually good, but CPU is fast enough and doesn't consume VRAM, so...
    device="cpu",
    hub="hf",
)


async def update_managed_scene_with_emotion(
    ws: simpleobsws.WebSocketClient, emotion: Emotion
):
    request = simpleobsws.Request("GetSceneList")
    response = await ws.call(request)
    if not response.ok():
        print("Could not get scene list while updating emotions")

    uuids = [scene["sceneUuid"] for scene in response.responseData["scenes"]]
    target_scene = SCENE_MAPPING[emotion]
    for uuid in uuids:
        request = simpleobsws.Request("GetSceneItemList", {"sceneUuid": uuid})
        response = await ws.call(request)
        if not response.ok():
            print(f"Could not get scene item list of {uuid}")
            continue

        sceneItems = response.responseData["sceneItems"]
        for sceneItem in sceneItems:
            sourceName = sceneItem["sourceName"]
            sourceId = sceneItem["sceneItemId"]

            if sourceName not in MANAGED_SCENES:
                continue

            request = simpleobsws.Request(
                "SetSceneItemEnabled",
                {
                    "sceneUuid": uuid,
                    "sceneItemId": sourceId,
                    "sceneItemEnabled": sourceName == target_scene,
                },
            )
            response = await ws.call(request)
            if not response.ok():
                print(f"Could not toggle scene item {sourceName}")


def get_emotion(frames: np.ndarray) -> dict[Emotion, float]:
    res = MODEL.generate(
        frames,
        input_len=len(frames),
        granularity="utterance",
        extract_embedding=False,
    )
    return {Emotion.from_index(idx): res[0]["scores"][idx] for idx in range(8)}


async def main():
    audio = pyaudio.PyAudio()
    stream = audio.open(
        format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK
    )

    print("Connecting to OBS...")
    password = os.getenv("OBS_PASSWORD")
    ws = simpleobsws.WebSocketClient(url="ws://localhost:4455", password=password)
    await ws.connect()
    await ws.wait_until_identified()

    history_frames = []

    # keep iterating through windows
    try:
        print("Listening to input...")
        while True:
            frames = []
            for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
                data = stream.read(CHUNK)
                frames.append(data)

            history_frames.append(np.frombuffer(b"".join(frames), dtype="float32"))
            frames_as_np = np.concat(history_frames)
            # print(max(frames_as_np))
            if max(frames_as_np) <= SILENCE_THRESHOLD:
                continue

            if len(history_frames) * RECORD_SECONDS >= WINDOW_SECONDS:
                history_frames.pop(0)

            emotions = get_emotion(frames_as_np)
            target_emotion, value = max(emotions.items(), key=lambda x: x[1])

            print(f"Guessed {target_emotion} with {value}")
            await update_managed_scene_with_emotion(ws, target_emotion)
    except KeyboardInterrupt:
        print("Quitting now...")
    finally:
        stream.stop_stream()
        stream.close()
        audio.terminate()
        print("Input stopped")
        await ws.disconnect()
        print("Disconnected from OBS")


if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main())
