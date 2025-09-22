import streamlit as st
import google.generativeai as genai
from googleapiclient.discovery import build
import os
import time
from dotenv import load_dotenv

# --------------------------
# --- Load environment variables ---
# --------------------------
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")   # Google Custom Search API key
GOOGLE_CSE_ID = os.getenv("GOOGLE_CSE_ID")     # Google Custom Search Engine ID

if not GEMINI_API_KEY or not GOOGLE_API_KEY or not GOOGLE_CSE_ID:
    st.error("❌ Please set GEMINI_API_KEY, GOOGLE_API_KEY, and GOOGLE_CSE_ID in your .env file.")
    st.stop()

# --------------------------
# --- Configure Gemini API ---
# --------------------------
genai.configure(api_key=GEMINI_API_KEY)
MODEL_PRIORITY = ["gemini-2.5-flash", "gemini-2.5-pro", "gemini-2.5-flash-lite"]

# --------------------------
# --- Google Search Function ---
# --------------------------
def google_search(query, num=3):
    try:
        service = build("customsearch", "v1", developerKey=GOOGLE_API_KEY)
        res = service.cse().list(q=query, cx=GOOGLE_CSE_ID, num=num).execute()
        results = []
        if "items" in res:
            for item in res["items"]:
                results.append({
                    "title": item.get("title"),
                    "link": item.get("link"),
                    "snippet": item.get("snippet")
                })
        return results
    except Exception as e:
        return [{"title": "Error", "link": "", "snippet": str(e)}]

# --------------------------
# --- Gemini Streaming Function ---
# --------------------------
def get_gemini_streaming(prompt):
    for model_name in MODEL_PRIORITY:
        try:
            model = genai.GenerativeModel(model_name)
            response = model.generate_content(prompt, stream=True)

            partial_text = ""
            for chunk in response:
                try:
                    if chunk.candidates:
                        for part in chunk.candidates[0].content.parts:
                            if hasattr(part, "text"):
                                partial_text += part.text
                                yield partial_text
                except Exception:
                    continue

            if partial_text.strip():
                return
            else:
                st.warning(f"⚠️ {model_name} returned no text. Trying next model...")
                continue

        except Exception as e:
            if "429" in str(e):
                st.warning(f"⏳ Rate limit hit on {model_name}. Trying next model...")
                time.sleep(2)
                continue
            else:
                st.error(f"⚠️ Error with {model_name}: {e}")
                continue

    yield "⚠️ Could not generate a response. All models failed."

# --------------------------
# --- Streamlit UI ---
# --------------------------
st.set_page_config(page_title="Gemini Chatbot", page_icon="🤖", layout="wide")
st.title("🤖Gemini Chatbot")
st.markdown("Chat with Gemini, get real-time Google results, and manage chats like ChatGPT.")

# --------------------------
# --- Session State ---
# --------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "past_chats" not in st.session_state:
    st.session_state.past_chats = []

# --------------------------
# --- Sidebar Actions ---
# --------------------------
with st.sidebar:
    st.header("Chat Options")
    
    if st.button("New Chat"):
        if st.session_state.messages:
            st.session_state.past_chats.append(st.session_state.messages.copy())
        st.session_state.messages = []

    if st.button("Show Past Chats"):
        st.subheader("Past Chats")
        for idx, chat in enumerate(st.session_state.past_chats[::-1], 1):
            st.markdown(f"### Chat {len(st.session_state.past_chats) - idx + 1}")
            for msg in chat:
                st.markdown(f"**{msg['role'].capitalize()}:** {msg['content']}")

    search_query = st.text_input("Search Past Chats")
    if search_query:
        st.subheader(f"Search Results for: {search_query}")
        for chat in st.session_state.past_chats[::-1]:
            filtered = [m for m in chat if search_query.lower() in m["content"].lower()]
            for msg in filtered:
                st.markdown(f"**{msg['role'].capitalize()}:** {msg['content']}")

# --------------------------
# --- Display Current Chat ---
# --------------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# --------------------------
# --- User Input ---
# --------------------------
if prompt := st.chat_input("Ask me anything..."):
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # --------------------------
    # --- Live Google Search + Gemini Reasoning ---
    # --------------------------
    if prompt.lower().startswith("search:"):
        query = prompt.replace("search:", "").strip()
        results = google_search(query, num=5)
        formatted_results = "\n".join([f"**{r['title']}**\n{r['link']}\n{r['snippet']}" for r in results])
        with st.chat_message("assistant"):
            st.markdown(f"🔎 **Search results for:** {query}\n\n{formatted_results}")
        st.session_state.messages.append({"role": "assistant", "content": formatted_results})
    else:
        # Build Gemini prompt using Google results for context
        search_results = google_search(prompt, num=5)
        context_text = "\n".join([f"- {r['title']}: {r['snippet']} ({r['link']})" for r in search_results])
        full_prompt = f"Answer the user query using the following context from recent web search results:\n{context_text}\n\nQuestion: {prompt}\nProvide a clear and concise answer, explaining any discrepancies if multiple sources differ."

        with st.chat_message("assistant"):
            placeholder = st.empty()
            streamed_text = ""
            for partial in get_gemini_streaming(full_prompt):
                streamed_text = partial
                placeholder.markdown(streamed_text + "▌")
            placeholder.markdown(streamed_text)

        st.session_state.messages.append({"role": "assistant", "content": streamed_text})
