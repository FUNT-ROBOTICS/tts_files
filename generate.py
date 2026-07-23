

import os
import re
import time
import logging
import requests
import pysbd
from pydub import AudioSegment
from concurrent.futures import ThreadPoolExecutor, as_completed

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

TEXT_FILE = os.path.join(BASE_DIR, "script.txt")

OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
CHUNKS_DIR = os.path.join(OUTPUT_DIR, "chunks")
FINAL_DIR = os.path.join(OUTPUT_DIR, "final")
LOGS_DIR = os.path.join(BASE_DIR, "logs")

MERGED_FILE = os.path.join(FINAL_DIR, "final_merged.wav")

API_URL = "http://127.0.0.1:8091/v1/audio/speech"
MODELS_URL = "http://127.0.0.1:8091/v1/models"

MODEL = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
VOICE = "sohee"
LANGUAGE = "English"

DEFAULT_INSTRUCTIONS = "Speak in a cheerful, energetic, and curious tone as if teaching an educational tutorial."
CHUNK_INSTRUCTIONS = {}

MAX_WORKERS = 16

TARGET_WORDS = 70
MIN_WORDS = 40
MAX_WORDS = 100

SILENCE_MS = 150
NORMALIZE_AUDIO = True
TARGET_DBFS = -20.0

CONNECT_TIMEOUT = 30
READ_TIMEOUT = 600
RETRY_STATUS = {429,500,502,503,504}

for d in (OUTPUT_DIR, CHUNKS_DIR, FINAL_DIR, LOGS_DIR):
    os.makedirs(d, exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("tts")
session = requests.Session()

def cleanup_chunks():
    for f in os.listdir(CHUNKS_DIR):
        if f.endswith(".wav"):
            os.remove(os.path.join(CHUNKS_DIR,f))

def check_server():
    try:
        r = session.get(MODELS_URL, timeout=(5, 10))
        return r.status_code == 200
    except Exception:
        return False
    
def count_words(text):
    return len(re.findall(r"\b[\w'-]+\b", text))

def split_into_chunks(text):
    seg=pysbd.Segmenter(language="en", clean=True)
    sents=seg.segment(text)
    chunks=[]; cur=[]; words=0
    for s in sents:
        s=s.strip()
        if not s: continue
        w=count_words(s)
        if not cur:
            cur=[s]; words=w; continue
        if words<MIN_WORDS:
            cur.append(s); words+=w; continue
        if words+w>MAX_WORDS:
            chunks.append(" ".join(cur)); cur=[s]; words=w; continue
        if abs(TARGET_WORDS-words)<=abs(TARGET_WORDS-(words+w)):
            chunks.append(" ".join(cur)); cur=[s]; words=w
        else:
            cur.append(s); words+=w
    if cur: chunks.append(" ".join(cur))
    return chunks

def post_with_retry(payload,retries=5):
    delay=1
    for i in range(retries):
        try:
            r=session.post(API_URL,json=payload,timeout=(CONNECT_TIMEOUT,READ_TIMEOUT))
            if r.status_code in RETRY_STATUS:
                raise requests.HTTPError(r.status_code)
            r.raise_for_status()
            return r.content
        except Exception:
            if i==retries-1: raise
            time.sleep(delay); delay*=2

def generate_chunk(idx,text,instructions):
    payload={"model":MODEL,"input":text,"voice":VOICE}
    if LANGUAGE: payload["language"]=LANGUAGE
    if instructions: payload["instructions"]=instructions
    audio=post_with_retry(payload)
    out=os.path.join(CHUNKS_DIR,f"chunk_{idx+1:03d}.wav")
    with open(out,"wb") as f: f.write(audio)
    return idx,out

def normalize(a):
    if not NORMALIZE_AUDIO or a.dBFS==float("-inf"): return a
    return a.apply_gain(TARGET_DBFS-a.dBFS)

def main():
    if not os.path.isfile(TEXT_FILE):
        log.error("script.txt not found"); return
    if not check_server():
     log.error("vLLM server is not running. Please start the vLLM server and try again.")
     return
    cleanup_chunks()
    raw=open(TEXT_FILE,"r",encoding="utf-8").read()
    chunks=split_into_chunks(raw)
    results={}; failed=[]
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futs={ex.submit(generate_chunk,i,c,CHUNK_INSTRUCTIONS.get(i,DEFAULT_INSTRUCTIONS)):(i,c) for i,c in enumerate(chunks)}
        for fut in as_completed(futs):
            i,c=futs[fut]
            try:
                _,p=fut.result(); results[i]=p
            except Exception as e:
                failed.append((i,str(e)))
    if failed:
        log.error("Generation aborted."); return
    merged=AudioSegment.empty()
    pause=AudioSegment.silent(duration=SILENCE_MS)
    for i in sorted(results):
        merged+=normalize(AudioSegment.from_wav(results[i]))
        if i!=max(results): merged+=pause
    merged.export(MERGED_FILE,format="wav")
    log.info(f"Saved: {MERGED_FILE}")

if __name__=="__main__":
    main()
