import threading
from configparser import ConfigParser
import glob
import random
import re
import pandas as pd
import requests
import time
import os
import shutil
import zipfile
from moviepy.editor import VideoFileClip, TextClip, CompositeVideoClip, AudioFileClip, ColorClip, concatenate_videoclips
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from flask_socketio import SocketIO, emit
from tqdm import tqdm
from elevenlabs.client import ElevenLabs
from traitlets import Bool
from dependencies.fonts import font_exists, install_fonts
from dependencies.imagemagick import install_imagemagick, is_imagemagick_installed
from moviepy.video.fx.all import crop
from dependencies.voices import VOICE_SETTINGS
from datetime import datetime
import csv

app = Flask(__name__)
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")

global VOICES
global DEV_MODE
start_time = datetime.now()

# region Variables config
# ! Edit voices here
VOICES = [
    "Bradley - Formal and Serious",
    "Drew",
]

# ! Script videos OUTPUT dir
# get desktop dir macos or windows whatever is os
if os.name == 'nt':
    SCRIPTS_OUTPUT_DIR = os.path.join(os.environ['USERPROFILE'], 'Desktop')
else:
    SCRIPTS_OUTPUT_DIR = os.path.join(os.environ['HOME'], 'Desktop')

# endregion

# region Loading config and setting things up
config_file = os.path.abspath('./dev_mode.ini' if os.path.exists('./dev_mode.ini') else './config.ini')
config = ConfigParser()
config.read(config_file)

no_of_parallel_executions = config.get('Setup', 'no_of_parallel_executions', fallback=20)

ELEVENLABS_API_KEY = config.get('Setup', 'elevenlabs_api_key', fallback=None)
if not ELEVENLABS_API_KEY:
    raise ValueError("elevenlabs_api_key not set")
else:
    client = ElevenLabs(api_key=ELEVENLABS_API_KEY)
# BATCH_SIZE = int(config.get('Setup', 'batch_size', fallback='100'))

INPUT_DIR = config.get('Setup', 'input_dir', fallback='input')
INPUT_FILE = config.get('Setup', 'input_hooks_csv', fallback='hooks.csv')
INPUT_FILE = os.path.join(INPUT_DIR, INPUT_FILE)

OUTPUT_DIR = config.get('Setup', 'output_dir', fallback='output')
OUTPUT_FILE = config.get('Setup', 'output_file', fallback='final_results.csv')

output_audios_folder = os.path.join(OUTPUT_DIR, 'audios')
output_videos_folder = os.path.join(OUTPUT_DIR, 'videos')
input_videos_folder = os.path.join(INPUT_DIR, 'video')
input_scripts_folder = os.path.join(INPUT_DIR, 'scripts')

OUT_VIDEO_DIM = config.get('Setup', 'output_video_dimensions', fallback='720x900')
OUT_VIDEO_HEIGHT = int(OUT_VIDEO_DIM.split('x')[1])
OUT_VIDEO_WIDTH = int(OUT_VIDEO_DIM.split('x')[0])

DEV_MODE = True if config.get('Setup', 'dev_mode', fallback='false').lower() == 'true' else False

if not is_imagemagick_installed():
    install_imagemagick()
    exit(1)
else:
    print("ImageMagick is installed.")

if not font_exists('mu.otf'):
    print("Installing fonts...")
    install_fonts()


def split_hook_text(hook_text):
    words = hook_text.split()
    hook_text = ' '.join(word.capitalize() for word in words)
    if ' - ' in hook_text:
        last_dash_index = hook_text.rfind('-')
        line1 = hook_text[:last_dash_index].strip()
        line2 = hook_text[last_dash_index + 1:].strip()
    else:
        line1 = hook_text
        line2 = ' Second LINE IS MISSING '
    return [line1, line2]


def cleanup_temp_files():
    for f in glob.glob("*temp*.mp3"):
        print("Removing temp file: " + f)
        os.remove(f)


def create_custom_text_clip(hook_text):
    hook_text = split_hook_text(hook_text)
    print(f"hook_text: {hook_text}")
    x_multiplier = OUT_VIDEO_WIDTH / 360
    y_multiplier = OUT_VIDEO_HEIGHT / 450
    min_red_area_h = int(round(75 * y_multiplier))
    min_white_area_h = int(round(30 * y_multiplier))
    fontsize1 = int(round(20 * x_multiplier))
    fontsize2 = int(round(14 * x_multiplier))

    color = {
        'white': (255, 255, 255),
        'red': (255, 0, 0),
        'black': (0, 0, 0),
        'light-black': (51, 51, 51),
        'shadow-white': (200, 200, 200),
    }

    x_margin = 5
    text_clip1 = TextClip(
        hook_text[0],
        size=(OUT_VIDEO_WIDTH - (x_margin * 2), 0),
        method='caption',
        font='dependencies/fonts/mu.otf',
        fontsize=fontsize1,
        color=f"rgb{color['white']}",
        align='center',
        stroke_color=f"rgb{color['shadow-white']}",
        stroke_width=1,
    )

    (text_clip1_w, text_clip1_h) = text_clip1.size
    if text_clip1_h > (min_red_area_h - 10):
        min_red_area_h = text_clip1_h + 10
    text_clip2 = TextClip(
        hook_text[1],
        size=(OUT_VIDEO_WIDTH - (x_margin * 2), 0),
        method='caption',
        font='dependencies/fonts/mu.otf',
        fontsize=fontsize2,
        color=f"rgb{color['black']}",
        align='center',
    )
    (text_clip2_w, text_clip2_h) = text_clip2.size
    if text_clip2_h > (min_white_area_h):
        min_white_area_h = text_clip2_h
    bg_clip1 = ColorClip(size=(OUT_VIDEO_WIDTH, min_red_area_h), color=color['red'])
    bg_clip2 = ColorClip(size=(OUT_VIDEO_WIDTH, min_white_area_h), color=color['white'])

    text_clip1_y_offset = ((min_red_area_h - text_clip1_h) / 2)
    text_clip2_y_offset = min_red_area_h + ((min_white_area_h - text_clip2_h) / 2)
    final_clip = CompositeVideoClip([
        bg_clip1.set_position((0, 0)),
        bg_clip2.set_position((0, min_red_area_h)),
        text_clip1.set_position((x_margin, text_clip1_y_offset)),
        text_clip2.set_position((x_margin, text_clip2_y_offset)),
    ], size=(OUT_VIDEO_WIDTH, OUT_VIDEO_HEIGHT))

    if DEV_MODE:
        try:
            print("final_clip size:", final_clip.size)
            print("bg_clip1 size:", bg_clip1.size)
            print("bg_clip2 size:", bg_clip2.size)
            print("text_clip1 size:", text_clip1.size)
            print("text_clip2 size:", text_clip2.size)
            print("bg_clip1 position at t=0:", bg_clip1.pos(0))
            print("bg_clip2 position at t=0:", bg_clip2.pos(0))
            print("text_clip1 position at t=0:", text_clip1.pos(0))
            print("text_clip2 position at t=0:", text_clip2.pos(0))
            sample_frame = final_clip.get_frame(0)
            print("Sample frame shape:", sample_frame.shape)
        except Exception as e:
            print("Error during frame validation:", e)
            input("Press Enter to continue...")
    return final_clip


def text_to_speech_file(text: str, save_file_path: str, remove_punctuation: bool = True) -> Bool:
    global VOICES
    if remove_punctuation:
        text = text.replace('-', ' ').replace('"', ' ').replace("'", ' ')
        text = re.sub(r'[^\w\s]', '', text)

    random_voice_name = random.choice(list(VOICES.keys()))

    response = client.text_to_speech.convert(
        voice_id=VOICES[random_voice_name],
        optimize_streaming_latency=2,
        output_format="mp3_22050_32",
        text=text,
        model_id="eleven_turbo_v2",
        voice_settings=VOICE_SETTINGS[random_voice_name],
    )

    with open(save_file_path, "wb") as f:
        for chunk in response:
            if chunk:
                f.write(chunk)

    return True, random_voice_name


def get_voices_ids(voice_names) -> dict:
    voices_url = 'https://api.elevenlabs.io/v1/voices'
    headers = {"xi-api-key": ELEVENLABS_API_KEY}
    while True:
        try:
            response = requests.get(voices_url, headers=headers)
            if response.status_code == 200:
                voices_data = response.json()
                VOICES = {}
                for voice in voices_data['voices']:
                    name = voice['name']
                    voice_id = voice['voice_id']

                    if name in voice_names:
                        VOICES[name] = voice_id
            return VOICES
        except Exception as e:
            print(f"Error getting voices IDs: {e}")
            for i in reversed(range(10)):
                print(f"Retrying in {i} seconds...", end="\r")
                time.sleep(1)


def process_audios(row, hook_number, hook_text, input_df, idx, voice_id):
    if row['Audio Filename'] in (None, '') or not os.path.exists(
            os.path.join(output_audios_folder, row['Audio Filename'])):
        print(f"Generating voiceover for hook {hook_number}...")
        audio_filename = os.path.join(output_audios_folder, f'hook_{hook_number}.mp3')
        status, voice_name = text_to_speech_file(hook_text, audio_filename)
        row['Voice'] = voice_name
        row['Audio Filename'] = os.path.basename(os.path.join(output_audios_folder, audio_filename))
        input_df.at[idx, 'Voice'] = voice_name
        input_df.at[idx, 'Audio Filename'] = row['Audio Filename']
        input_df.to_csv(INPUT_FILE, index=False, quotechar='"', quoting=csv.QUOTE_ALL, escapechar='\\')


def process_audio_on_videos(row, video_files, idx, input_df, hook_number, hook_text, num_videos_to_use, audio_clip):
    print('===========================')
    print(f'idx={idx}')
    print(f'audio_clip={audio_clip}')
    print(f'num_videos_to_use={num_videos_to_use}')
    print(f'video_files_to_use={video_files}')
    print('===========================')

    considered_videos = [os.path.join(input_videos_folder, video_files[i]) for i in range(num_videos_to_use)]
    row['Input Video Filename'] = [os.path.basename(considered_video) for considered_video in considered_videos]
    input_df.at[idx, 'Input Video Filename'] = row['Input Video Filename']
    input_df.to_csv(INPUT_FILE, index=False, quotechar='"', quoting=csv.QUOTE_ALL)

    print(f"Generating final hook {hook_number} video...")

    each_video_duration = audio_clip.duration / num_videos_to_use
    video_clips = []
    for considered_vid in considered_videos:
        video_clip = VideoFileClip(considered_vid).resize(width=OUT_VIDEO_WIDTH)
        (w, h) = video_clip.size
        cropped_clip = crop(video_clip.subclip(0, each_video_duration), width=OUT_VIDEO_WIDTH,
                            height=OUT_VIDEO_HEIGHT, x_center=w / 2, y_center=h / 2)
        video_clips.append(cropped_clip)

    final_video_clip = concatenate_videoclips(video_clips)

    custom_text_clip = create_custom_text_clip(hook_text)

    final_clip = CompositeVideoClip([
        final_video_clip.audio_fadein(0.2).audio_fadeout(0.2),
        final_video_clip,
        custom_text_clip
    ]).set_audio(audio_clip).set_duration(audio_clip.duration)

    output_video_filename = os.path.join(output_videos_folder, f'hook_{idx}.mp4')
    final_clip.write_videofile(output_video_filename, temp_audiofile=f"temp-audio_{idx}.m4a", remove_temp=True,
                               codec='libx264', audio_codec="aac")

    input_df.at[idx, 'Hook Video Filename'] = os.path.basename(output_video_filename)
    input_df.to_csv(INPUT_FILE, index=False, quotechar='"', quoting=csv.QUOTE_ALL, escapechar='\\')
    cleanup_temp_files()


def process_script_file(hook, script, temp_filename, final_filename, idy, idx):
    hook_clip = VideoFileClip(os.path.join(output_videos_folder, hook))
    script_clip = VideoFileClip(os.path.join(input_scripts_folder, script)).resize(width=OUT_VIDEO_WIDTH)
    (w, h) = script_clip.size
    script_clip = crop(script_clip, width=OUT_VIDEO_WIDTH, height=OUT_VIDEO_HEIGHT, x_center=w / 2, y_center=h / 2)

    final_clip = concatenate_videoclips([hook_clip, script_clip])
    final_clip.write_videofile(temp_filename, temp_audiofile=f"temp-audio_{idx}_{idy}.m4a", remove_temp=True,
                               codec='libx264', audio_codec="aac")

    print(f'<<<<<<<<<<<<<<<<<<<<<<<<<<moving {temp_filename} to  {final_filename}')
    shutil.move(temp_filename, final_filename)


def calculate_total_hours():
    try:
        end_time = datetime.now()
        time_difference = end_time - start_time
        total_seconds = time_difference.total_seconds()
        total_hours = total_seconds / 3600
        print(f'total_hours={total_hours}')
    except Exception as e:
        print('issue with duration calculation')
        print(e)


def process(INPUT_DIR, INPUT_FILE, OUTPUT_DIR, voice_id):
    input_videos_folder = os.path.join(INPUT_DIR, 'video')
    input_scripts_folder = os.path.join(INPUT_DIR, 'scripts')
    output_audios_folder = os.path.join(OUTPUT_DIR, 'audios')
    output_videos_folder = os.path.join(OUTPUT_DIR, 'videos')
    global VOICES

    SCRIPTS_OUTPUT_DIR = os.path.join(os.getcwd(), 'static', 'output_root', 'final')
    print(f'creating script folder path={SCRIPTS_OUTPUT_DIR}')
    script_folder = SCRIPTS_OUTPUT_DIR
    os.makedirs(script_folder, exist_ok=True)
    print('script path created')

    if not os.path.exists(INPUT_FILE):
        raise Exception(f"Input file {INPUT_FILE} does not exist")

    if len(os.listdir(input_videos_folder)) == 0:
        raise Exception(f"input/videos folder {input_videos_folder} does not contain any videos")

    video_files = sorted([f for f in os.listdir(input_videos_folder) if f.endswith('.mp4')])

    print(f'video_files={video_files}')

    input_df = pd.read_csv(INPUT_FILE)
    for col in ["Hook Video Filename", "Input Video Filename", "Audio Filename", "Voice"]:
        if col not in input_df.columns:
            input_df[col] = ''
    input_df = input_df[input_df['Hook Text'] != '']
    input_df = input_df.fillna('')

    l_unprocessed_rows = len(input_df[input_df['Hook Video Filename'] == ''])
    if l_unprocessed_rows == 0:
        print("No unprocessed rows found in csv file.")

    print(f"Found total {len(input_df)} valid rows in csv file.")
    print(f"Total input videos: {len(video_files)}")
    print(f"Unprocessed rows: {l_unprocessed_rows}")

    VOICES = get_voices_ids(VOICES)
    all_hooks = []

    for idx_1, row in tqdm(input_df.iterrows(), total=len(input_df), desc="Processing rows"):
        hook_text = row['Hook Text']
        hook_number = idx_1 + 1
        process_audios(row, hook_number, hook_text, input_df, idx_1, voice_id)

    current_thread_count = 0
    for idx, row in tqdm(input_df.iterrows(), total=len(input_df), desc="Processing rows"):
        hook_text = row['Hook Text']
        hook_number = idx + 1

        if row['Hook Video Filename'] != '' and os.path.exists(
                os.path.join(output_videos_folder, row['Hook Video Filename'])):
            continue

        print('')
        audio_clip = AudioFileClip(os.path.join(output_audios_folder, row['Audio Filename']))
        video_index = idx % len(video_files)
        num_videos_to_use = int(round(audio_clip.duration / 2))

        video_file_size = len(video_files)
        if num_videos_to_use + video_index > video_file_size:
            num_videos_to_use = video_file_size - video_index

        last_video = video_index + num_videos_to_use
        video_files_to_use = video_files[video_index:last_video]

        hook_job = threading.Thread(target=process_audio_on_videos, args=(row, video_files_to_use, idx, input_df, hook_number, hook_text, num_videos_to_use, audio_clip))
        hook_job.start()
        all_hooks.append(hook_job)
        current_thread_count += 1
        if current_thread_count == int(no_of_parallel_executions):
            for hook in all_hooks:
                hook.join()
            all_hooks.clear()
            current_thread_count = 0
        else:
            all_hooks.append(hook_job)

    for hook in all_hooks:
        hook.join()

    print("Generating final output script videos...")

    script_files = sorted([f for f in os.listdir(input_scripts_folder) if f.endswith('.mp4')])
    hook_files = sorted([f for f in os.listdir(output_videos_folder) if f.endswith('.mp4')])

    current_thread_count = 0
    for idx, script in enumerate(script_files):
        for idy, hook in enumerate(tqdm(hook_files, desc=f"Processing Script {idx + 1} hooks")):
            hook_script_filename = f"{os.path.splitext(hook)[0]}_{os.path.splitext(script)[0]}.mp4".replace(" ", "_")
            temp_filename = os.path.join(script_folder, f"temp_{idx}_{idy}.mp4")
            final_filename = os.path.join(script_folder, hook_script_filename)
            if os.path.isfile(final_filename):
                continue

            hook_job = threading.Thread(target=process_script_file,
                                        args=(hook, script, temp_filename, final_filename, idy, idx))
            hook_job.start()
            all_hooks.append(hook_job)
            current_thread_count += 1
            if current_thread_count == int(no_of_parallel_executions):
                for hook in all_hooks:
                    hook.join()
                all_hooks.clear()
                current_thread_count = 0
            else:
                all_hooks.append(hook_job)

        for hook in all_hooks:
            hook.join()

    calculate_total_hours()

    # Create a ZIP archive of the output files
    output_zip = os.path.join(OUTPUT_DIR, 'output_files.zip')
    with zipfile.ZipFile(output_zip, 'w') as zipf:
        for root, dirs, files in os.walk(OUTPUT_DIR):
            for file in files:
                zipf.write(os.path.join(root, file), os.path.relpath(os.path.join(root, file), OUTPUT_DIR))

    # Clean up input and output files
    shutil.rmtree(INPUT_DIR)
    shutil.rmtree(OUTPUT_DIR)

    print('!!!!!!!!! PROCESS COMPLETED !!!!!!!!!!')
    return output_zip


@app.route('/process', methods=['POST'])
def handle_process():
    try:
        data = request.json
        input_dir = data['input_dir']
        input_file = data['input_file']
        output_dir = data['output_dir']
        voice_id = data['voice_id']

        output_zip = process(input_dir, input_file, output_dir, voice_id)
        return send_file(output_zip, as_attachment=True)

    except Exception as e:
        print(f"Error processing request: {e}")
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    socketio.run(app, debug=True, host='0.0.0.0')
