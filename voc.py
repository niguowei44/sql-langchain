import speech_recognition as sr  # 监听麦克风输入，实现语音转文本的功能
r = sr.Recognizer()
with sr.Microphone() as source:
    r.adjust_for_ambient_noise(source, duration=1)
    audio = r.listen(source, timeout=5, phrase_time_limit=30)
    audio_text = r.recognize_whisper(audio, language="chinese")
print(audio_text)
