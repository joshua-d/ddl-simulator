import pyttsx3

tts_engine = pyttsx3.init()
tts_engine.setProperty('rate', 200)

voices = tts_engine.getProperty('voices')
tts_engine.setProperty('voice', voices[1].id)

def tts(text):
    tts_engine.say(text)
    tts_engine.runAndWait()