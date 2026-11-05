"""Step 1: LLM basics - tokens, temperature, hallucination.

Run:  .venv\\Scripts\\python step01_llm_basics.py
"""
import os

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()  # .env file se GOOGLE_API_KEY padhta hai

MODEL = os.getenv("GEMINI_CHAT_MODEL", "gemini-3.5-flash-lite")


def make_llm(temperature: float) -> ChatGoogleGenerativeAI:
    # temperature: 0 = lagbhag same jawab har baar (legal ke liye best)
    #              1+ = zyada random/creative
    # max_retries: 503 "high demand" jaise temporary errors par auto-retry
    return ChatGoogleGenerativeAI(
        model=MODEL, temperature=temperature, max_retries=2
    )


# ---- Experiment 1: tokens ------------------------------------------------
# LLM text nahi, "tokens" padhta hai. Cost aur context limit tokens me hoti hai.
llm = make_llm(0)
resp = llm.invoke("DPDP Act 2023 ka full form kya hai? Ek line me batao.")
print("ANSWER:", resp.text)
print("TOKEN USAGE:", resp.usage_metadata)  # input_tokens, output_tokens, total

# ---- Experiment 2: temperature -------------------------------------------
prompt = "Ek line me: data fiduciary kya hota hai?"
for t in (0, 1.5):
    print(f"\n--- temperature={t} (3 runs) ---")
    for _ in range(3):
        print("-", make_llm(t).invoke(prompt).text.strip())

# ---- Experiment 3: hallucination -----------------------------------------
# Model ko document diye bina specific section poocho. Dekho kya hota hai.
q = "IT Rules 2021 ke Rule 4(1)(a) me exactly kya likha hai? Word-by-word batao."
print("\n--- Hallucination test (no context given) ---")
print(make_llm(0).invoke(q).text)
