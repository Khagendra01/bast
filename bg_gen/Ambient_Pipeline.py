from .LLMPromptGenerator import LLMPromptGenerator
from .GenMusicFromPrompt import GenMusicFromPrompt
from .LLMPromptConstraints import MusicGenInfo

import os
import requests
import tempfile
import torchaudio
import torch
import numpy as np
import json

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Parameters for Audio Chunking and Music Duration Generation:
DFLT_CHUNK_LEN_S = 15 # For Whisper
DESIRED_CHUNK_LEN = 15 # For grouping whisper chunks (desired length of a chunk)
MAX_CHUNK_LEN = 20 # For grouping whisper chunks (max length of a chunk)

GROUP_WORD_COUNT = 60 #120 
SONG_DUR_SECONDS = 30 #60 
PREV_SONG_DUR = 2 # 4

MAX_GROUP_CNT = 3

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

class Music_Gen_Pipeline():
    
    def __init__(self,
                audio_pipe=None,
                # audio_model=None,
                # audio_processor=None,
                extractor=None,
                generator=None,
                # song_dur_seconds=SONG_DUR_SECONDS,
                # previous_song_duration=PREV_SONG_DUR,
                device="cuda",
                verbose=True,
                desired_section_size=GROUP_WORD_COUNT,
                openai_api_key=None) -> None:
        # self.audio_model = audio_model
        # self.audio_processor = audio_processor
        self.audio_pipe = audio_pipe
        # self.audio_device = 'cpu'
        self.audio_device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        
        self.extractor = extractor
        self.generator = generator
        
        # self.previous_song_duration = previous_song_duration
        self.device = device
        self.verbose = verbose
        self.desired_section_size = desired_section_size
        self.original_audio = None
        self.openai_api_key = openai_api_key
        
        if self.extractor is None:
            self.extractor = LLMPromptGenerator() #device=self.device)
            if self.verbose:
                print("Extractor not provided, using default")
        if self.generator is None:
            self.generator = GenMusicFromPrompt(device=self.device)
            if self.verbose:
                print("Generator not provided, using default")
    
    
    def load_txt(self, file_path):
        with open(file_path) as f: 
            book_text = f.read()
        return book_text
                
    def text_to_sections(self, text, desired_section_size=None, max_group_count=None):
        
        # If the input is a file path, load the text from the file
        if os.path.exists(text):
            text = self.load_txt(text)
        
        words = text.split()
        if desired_section_size is None:
            desired_section_size = self.desired_section_size

        # Calculate the total number of words
        total_words = len(words)

        # Determine the number of sections, aiming for equally sized sections
        # Calculate the optimal number of sections to avoid a significantly shorter final section
        optimal_num_sections = round(total_words / desired_section_size)

        # Calculate the new section size to more evenly distribute words across sections
        new_section_size = total_words // optimal_num_sections if total_words % optimal_num_sections == 0 else (total_words // optimal_num_sections) + 1

        # Adjust the last section to avoid being too short
        if total_words % new_section_size < new_section_size / 2:
            optimal_num_sections += 1

        word_sections = [' '.join(words[i:i+new_section_size]) for i in range(0, total_words, new_section_size)]
        
        if max_group_count is not None and len(word_sections) > max_group_count:
            word_sections = word_sections[:max_group_count]
        
        return word_sections
    
    def transcribe_with_openai(self, audio_path, api_key, model="whisper-1"):
        url = "https://api.openai.com/v1/audio/transcriptions"
        headers = {"Authorization": f"Bearer {api_key}"}
        with open(audio_path, "rb") as audio_file:
            files = {"file": audio_file}
            data = {"model": model, "response_format": "verbose_json"}
            response = requests.post(url, headers=headers, files=files, data=data)
        response.raise_for_status()
        return response.json()

    def audio_to_sections(self, wav_audio, default_chunk_length_s=30, desired_lengths=20, max_length=30, last_chunk_buffer=0, openai_api_key=None):
        """
        Use OpenAI Whisper API for ASR instead of local pipeline.
        """
        def group_chunks(chunks, desired_lengths=20, max_length=30):
            def duration(chunk):
                return chunk['timestamp'][1] - chunk['timestamp'][0]
            new_chunks = []
            current_chunk = None
            for chunk in chunks:
                if current_chunk is not None:
                    new_duration = duration(current_chunk) + duration(chunk)
                    if new_duration > max_length:
                        new_chunks.append(current_chunk)
                        current_chunk = None
                    else:
                        current_chunk['timestamp'][1] = chunk['timestamp'][1]
                        current_chunk['duration'] = new_duration
                        current_chunk['text'] += chunk['text']
                        if new_duration > desired_lengths and current_chunk['text'][-1] in ['.', '!', '?', '\n']:
                            new_chunks.append(current_chunk)
                            current_chunk = None
                            continue
                if current_chunk is None:
                    current_chunk = chunk
                    current_chunk['timestamp'] = list(chunk['timestamp'])
                    current_chunk['duration'] = duration(current_chunk)
                    continue
            if current_chunk is not None:
                new_chunks.append(current_chunk)
            total_duration = sum([c['duration'] for c in new_chunks[:-1]])
            new_chunks[-1]['duration'] = ((new_chunks[-1]['timestamp'][1]) - total_duration) + last_chunk_buffer
            print('new chunks', new_chunks)
            return new_chunks
        print(wav_audio.keys())
        if 'duration' not in wav_audio:
            wav_audio['duration'] = wav_audio['array'].shape[0] / wav_audio['sampling_rate']
            print('duration calculated from array', wav_audio['duration'])
        # Save audio to temp file for OpenAI API
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            arr = wav_audio['array']
            if arr.ndim == 1:
                arr = arr[None, :]  # Add channel dimension for mono
            torchaudio.save(tmp.name, torch.tensor(arr), wav_audio['sampling_rate'])
            tmp_path = tmp.name
        api_key = openai_api_key or self.openai_api_key or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API key must be provided via argument, class, or environment variable.")
        print('Calling OpenAI Whisper API for transcription...')
        result = self.transcribe_with_openai(tmp_path, api_key)
        # Parse result for chunks and text
        # OpenAI returns 'segments' with 'start', 'end', 'text'
        if 'segments' in result:
            chunks = []
            for seg in result['segments']:
                chunk = {
                    'timestamp': [seg['start'], seg['end']],
                    'text': seg['text'],
                }
                chunks.append(chunk)
            text = result.get('text', ' '.join([c['text'] for c in chunks]))
        else:
            # fallback: treat all as one chunk
            chunks = [{
                'timestamp': [0, wav_audio['duration']],
                'text': result.get('text', ''),
            }]
            text = result.get('text', '')
        print('Transcription result:', text)
        chunks = group_chunks(chunks, desired_lengths=desired_lengths, max_length=max_length)
        return chunks, text
        
    def sections_to_prompts(self, word_sections, flush_extractor=False, verbose=None, **kwargs):
        verbose = verbose if verbose is not None else self.verbose
        
        print(word_sections)
        
        # Extract JSON Info
        prompts, durations= self.extractor.generate_from_chunks(word_sections, verbose=verbose, **kwargs)
        
        
        return prompts, durations
    
    def prompts_to_music(self, prompts, durations, **kwargs):
        
        music = self.generator.generate_from_list(prompts, durations, **kwargs)
        
        save_file_loc = 'music.wav'
        if save_file_loc is not None:
            if os.path.isdir(save_file_loc):                
                base_name = os.path.basename(save_file_loc).split('.')[0] + '.wav'
                save_file_loc = os.path.join(save_file_loc, base_name)
            self.generator.save_audio(save_file_loc)
    
        
    # def generate_music(self, text, song_dur_seconds=SONG_DUR_SECONDS):
    def audio_to_music(self, wav_audio, song_dur_seconds=SONG_DUR_SECONDS, previous_song_duration=PREV_SONG_DUR, **kwargs):
        # if wav_audio is a np.ndarray (ie. from an mp3 file), "sampling_rate" must be provided in kwargs.
        
        path = None
        if isinstance(wav_audio, str):
            # print('loading from audio not supported yet')
            path = wav_audio
            raise NotImplementedError('loading from audio not supported yet')
            # raise Exception('loading from audio not supported yet')
        # if "is_mp3" in kwargs and kwargs['is_mp3']:
        elif isinstance(wav_audio, np.ndarray):
            if 'sampling_rate' not in kwargs:
                raise ValueError('sampling_rate must be provided for numpy array audio')
            wav_audio = {'path': path, 'array': wav_audio, 'sampling_rate': kwargs['sampling_rate']}
            # 'path', 'array', 'sampling_rate'
              
            # wav_audio = convert_audio(wav_audio, 'wav')
        
        if 'sampling_rate' not in wav_audio:
            if 'sampling_rate' in kwargs:
                wav_audio['sampling_rate'] = kwargs['sampling_rate']
            elif 'sample_rate' in kwargs:
                wav_audio['sampling_rate'] = kwargs['sample_rate']
            else:
                raise ValueError('sampling_rate must be provided in some way')
        
            
        
        # Extract JSON Info
        print("Splitting Audio to chunks")
        chunks, text = self.audio_to_sections(wav_audio, 
                                              default_chunk_length_s=DFLT_CHUNK_LEN_S,
                                              max_length=MAX_CHUNK_LEN,
                                              desired_lengths=DESIRED_CHUNK_LEN)


        print("Extracting Info from chunks")
        prompts, durations = self.sections_to_prompts(chunks, **kwargs)
        print(prompts)
        
        save_info_loc = kwargs.get('save_e_info_loc', None)
        if save_info_loc is not None:
            # info_preped = [p.__dict__ for p in info]
            # info_preped = [p.to_dict() for p in self.extractor.info]
            info_preped = self.extractor.info
            
            prepped_output = {'chunks': chunks, 
                              'info': info_preped,
                              'prompts': prompts, 
                              'durations': durations, 
                              }
            
            with open(save_info_loc, 'w') as f:
                # json.dump(info_preped, f)
                json.dump(prepped_output, f)
                
        
        print("Generating Music from prompts")
        
        
        self.prompts_to_music(prompts, durations, **kwargs)
        
        return self.generator.song, self.generator.sample_rate
        

    def get_chunks(self, text, **kwargs):
        # Parameters
        min_words = kwargs.get('min_words', 50)
        min_duration = kwargs.get('min_duration', 20.0)
        add_sec_per_2words = kwargs.get('add_sec_per_2words', 1.0)


        # # For testing
        # min_words = kwargs.get('min_words', 100)
        # min_duration = kwargs.get('min_duration', 40.0)
        # add_sec_per_2words = kwargs.get('add_sec_per_2words', 2)

        print(text)
        # If the input is a file path, load the text from the file
        if os.path.exists(text):
            text = self.load_txt(text)
        words = text.split()
        n = len(words)
        chunks = []
        i = 0
        start_time = 0.0
        while i < n:
            # Start with at least min_words
            end_idx = i + min_words
            if end_idx > n:
                end_idx = n
            # Extend to next period after min_words
            while end_idx < n and words[end_idx-1][-1] != '.':
                end_idx += 1
            # If still not at a period, just go to n
            if end_idx > n:
                end_idx = n
            chunk_words = words[i:end_idx]
            chunk_text = ' '.join(chunk_words)
            # Calculate duration
            extra_words = max(0, len(chunk_words) - min_words)
            extra_secs = (extra_words // 2) * add_sec_per_2words
            duration = min_duration + extra_secs
            end_time = start_time + duration
            chunk = {
                'timestamp': [start_time, end_time],
                'text': chunk_text,
                'duration': duration
            }
            chunks.append(chunk)
            start_time = end_time
            i = end_idx
        return chunks

    def text_to_music(self, text, **kwargs):

        chunks = self.get_chunks(text, **kwargs)
        prompts, durations = self.sections_to_prompts(chunks, **kwargs)
        print(prompts)
        
        save_info_loc = kwargs.get('save_e_info_loc', None)
        if save_info_loc is not None:
            # info_preped = [p.__dict__ for p in info]
            # info_preped = [p.to_dict() for p in self.extractor.info]
            info_preped = self.extractor.info
            
            prepped_output = {'chunks': chunks, 
                              'info': info_preped,
                              'prompts': prompts, 
                              'durations': durations, 
                              }
            
            with open(save_info_loc, 'w') as f:
                # json.dump(info_preped, f)
                json.dump(prepped_output, f)
                
        
        print("Generating Music from prompts")
        
        
        self.prompts_to_music(prompts, durations, **kwargs)
        
        return self.generator.song, self.generator.sample_rate




# Always load OpenAI API key from environment if not provided
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if OPENAI_API_KEY:
    print("[Ambient_Pipeline] Loaded OpenAI API key from environment.")
else:
    print("[Ambient_Pipeline] WARNING: No OpenAI API key found in environment!")

extractor = LLMPromptGenerator(openai_api_key=OPENAI_API_KEY)

generator = GenMusicFromPrompt(device=device)

pipe = Music_Gen_Pipeline(extractor=extractor, generator=generator, device=device, verbose=True, openai_api_key=OPENAI_API_KEY)



def text_to_music(text,**kwargs):
    if 'flush' not in kwargs:
        kwargs['flush'] = True
    #song, song_sr = pipe.audio_to_music(audio, **kwargs)
    song, song_sr = pipe.text_to_music(text, **kwargs)
    return song, song_sr

