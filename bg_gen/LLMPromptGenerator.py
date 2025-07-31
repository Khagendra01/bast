import json
import re
import requests
import os
from .LLMPromptConstraints import MusicGenInfo

# Add dotenv support
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv not installed, skip

class LLMPromptGenerator:
    def __init__(self, openai_api_key=None, model_name="gpt-4o"):
        # Always try to load from env if not provided
        if openai_api_key is None:
            openai_api_key = os.getenv("OPENAI_API_KEY")
            if openai_api_key:
                print("[LLMPromptGenerator] Loaded OpenAI API key from environment.")
            else:
                print("[LLMPromptGenerator] WARNING: No OpenAI API key found in environment or arguments!")
        else:
            print("[LLMPromptGenerator] OpenAI API key provided via argument.")
        self.model_name = model_name
        self.openai_api_key = openai_api_key
        self.info = []
        self.prompts = []
        self.long_term_prompt = None
        self.major_key = None
        self.instrument = None

    def call_openai_gpt(self, prompt, api_key=None, model=None):
        api_key = api_key or self.openai_api_key
        model = model or self.model_name
        url = "https://api.openai.com/v1/chat/completions"
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        data = {
            "model": model,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant that outputs only valid JSON objects."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.7,
            "max_tokens": 512
        }
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        result = response.json()
        return result["choices"][0]["message"]["content"]

    def generate_llm_prompt(self, text, prompt=None):
        if prompt is None:
            prompt = (
                "TEXT: {}\n\n"
                "TASK: In JSON Format, a piece of music generated as background ambience for the above text would have these qualities: "
                "(tone, intensity, setting, tempo, musical_instrument, is_major_key). "
                "Respond ONLY with a valid JSON object."
            )
        return prompt.format(text)

    def extract_json_from_llm_output(self, llm_output):
        pattern = r'\{.*?\}'  # Non-greedy match for JSON-like objects
        matches = re.findall(pattern, llm_output, re.DOTALL)
        if not matches:
            return {}
        longest_match = ""
        for match in matches:
            if len(match) > len(longest_match):
                try:
                    json.loads(match)
                    longest_match = match
                except json.JSONDecodeError:
                    continue
        if longest_match:
            try:
                json_obj = json.loads(longest_match)
                return json_obj
            except json.JSONDecodeError:
                return {}
        else:
            return {}

    def generate_musicgen_prompt(self, text, music_gen_info=MusicGenInfo, prompt=None, flush=False):
        llm_prompt = self.generate_llm_prompt(text, prompt=prompt)
        llm_output = self.call_openai_gpt(llm_prompt)
        music_attributes = self.extract_json_from_llm_output(llm_output)
        if flush or self.major_key is None:
            self.major_key = music_attributes.get('is_major_key', True)
            self.instrument = music_attributes.get('musical_instrument', 'piano')
        prompt = (
            f"Ambient Background music with a {music_attributes.get('tone', 'neutral')} tone and {music_attributes.get('intensity', 'medium')} intensity, "
            f"using {self.instrument} instrumentation to create a {music_attributes.get('setting', 'general')} setting. "
            f"The piece moves at a {music_attributes.get('tempo', 'moderate')} pace, "
            f"{'in a major key' if self.major_key else 'in a minor key'}, "
            "evoking an immersive atmosphere."
        )
        self.info.append(music_attributes)
        self.prompts.append(prompt)
        return prompt

    def generate_from_chunks(self, chunks, music_gen_info=MusicGenInfo, prompt=None, **kwargs):
        prompts = []
        durations = []
        flush = False
        if 'flush' in kwargs:
            flush = kwargs['flush']
        if 'flush_extractor' in kwargs:
            flush = kwargs['flush_extractor']
        if flush:
            self.prompts = []
            self.info = []
        for chunk in chunks:
            prompts.append(self.generate_musicgen_prompt(chunk['text'], flush=flush))
            durations.append(float(chunk['duration']))
        return prompts, durations
