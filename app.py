import logging
import eventlet

# Now monkey-patch with eventlet
eventlet.monkey_patch()

import tempfile
import signal
import os
import shutil
from flask import Flask, request, jsonify, send_file, Response
from flask_cors import CORS
from flask_socketio import SocketIO, emit
import threading
import pandas as pd
import glob
import re
import requests
from moviepy.editor import VideoFileClip, TextClip, CompositeVideoClip, AudioFileClip, ColorClip, concatenate_videoclips
from tqdm import tqdm
from datetime import datetime
from moviepy.video.fx.all import crop
import csv
import zipfile
from threading import Lock

app = Flask(__name__)
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")

upload_tasks = {}
tasks_lock = Lock()
process_pids = {}
canceled_tasks = set()

start_time = datetime.now()

# Define a constant path for the output zip file
OUTPUT_ZIP_PATH = 'output.zip'

def delete_temp_dir(temp_dir):
    try:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        print(f"Temporary directory {temp_dir} deleted successfully.")
    except Exception as e:
        print(f"Error deleting temporary directory {temp_dir}: {str(e)}")

@app.route('/process', methods=['POST'])
def process_endpoint():
    task_id = request.form.get('task_id')
    temp_dir = tempfile.mkdtemp(prefix=f"task_{task_id}_")

    return process_files(temp_dir, task_id)

def process_files(temp_dir, task_id):
    script_file = request.files.get('script')
    video_files = request.files.getlist('video')
    input_csv_file = request.files.get('input_csv')
    voice_id = request.form.get('voice_id')
    api_key = request.form.get('api_key')
    parallel_processing = request.form.get('parallel_processing')

    if not script_file or not video_files or not input_csv_file or not voice_id or not api_key or not parallel_processing:
        return jsonify({"error": "Missing form data"}), 400

    input_videos_folder = os.path.join(temp_dir, 'input', 'video')
    input_scripts_folder = os.path.join(temp_dir, 'input', 'scripts')
    output_audios_folder = os.path.join(temp_dir, 'output', 'audios')
    output_videos_folder = os.path.join(temp_dir, 'output', 'videos')

    os.makedirs(input_videos_folder, exist_ok=True)
    os.makedirs(input_scripts_folder, exist_ok=True)
    os.makedirs(output_audios_folder, exist_ok=True)
    os.makedirs(output_videos_folder, exist_ok=True)

    script_file_path = os.path.join(input_scripts_folder, script_file.filename)
    script_file.save(script_file_path)

    video_files_paths = []
    for video_file in video_files:
        video_file_path = os.path.join(input_videos_folder, video_file.filename)
        video_file.save(video_file_path)
        video_files_paths.append(video_file_path)

    input_csv_file_path = os.path.join(temp_dir, 'input', input_csv_file.filename)
    input_csv_file.save(input_csv_file_path)

    params = {
        "input_dir": os.path.join(temp_dir, 'input'),
        "output_dir": os.path.join(temp_dir, 'output'),
        "script_file_path": script_file_path,
        "video_files_paths": video_files_paths,
        "input_csv_file_path": input_csv_file_path,
        "voice_id": voice_id,
        "api_key": api_key,
        "parallel_processing": parallel_processing,
        "task_id": task_id,
        "temp_dir": temp_dir
    }

    socketio.start_background_task(target=process, params=params)
    return jsonify({"message": "Processing started", "task_id": task_id})

def process(params):
    try:
        global ELEVENLABS_API_KEY, no_of_parallel_executions

        ELEVENLABS_API_KEY = params['api_key']
        no_of_parallel_executions = params['parallel_processing']

        INPUT_DIR = params['input_dir']
        INPUT_FILE = params['input_csv_file_path']
        OUTPUT_DIR = params['output_dir']
        voice_id = params['voice_id']
        task_id = params['task_id']
        temp_dir = params['temp_dir']

        input_videos_folder = os.path.join(INPUT_DIR, 'video')
        input_scripts_folder = os.path.join(INPUT_DIR, 'scripts')
        output_audios_folder = os.path.join(OUTPUT_DIR, 'audios')
        output_videos_folder = os.path.join(OUTPUT_DIR, 'videos')

        OUT_VIDEO_DIM = '720x900'
        OUT_VIDEO_HEIGHT = int(OUT_VIDEO_DIM.split('x')[1])
        OUT_VIDEO_WIDTH = int(OUT_VIDEO_DIM.split('x')[0])

        SCRIPTS_OUTPUT_DIR = os.path.join(OUTPUT_DIR, 'final')
        os.makedirs(SCRIPTS_OUTPUT_DIR, exist_ok=True)

        if not os.path.exists(INPUT_FILE):
            raise Exception(f"Input file {INPUT_FILE} does not exist")

        if len(os.listdir(input_videos_folder)) == 0:
            raise Exception(f"input/videos folder {input_videos_folder} does not contain any videos")

        video_files = sorted([f for f in os.listdir(input_videos_folder) if f.endswith('.mp4')])

        input_df = pd.read_csv(INPUT_FILE)
        for col in ["Hook Video Filename", "Input Video Filename", "Audio Filename", "Voice"]:
            if col not in input_df.columns:
                input_df[col] = ''
        input_df = input_df[input_df['Hook Text'] != '']
        input_df = input_df.fillna('')

        l_unprocessed_rows = len(input_df[input_df['Hook Video Filename'] == ''])
        if l_unprocessed_rows == 0:
            print("No unprocessed rows found in csv file.")

        all_hooks = []
        total_rows = len(input_df)
        current_row = 0

        # Emit initial progress step
        socketio.emit('progress', {'task_id': task_id, 'progress': 0, 'step': f"0/{total_rows}"})

        for idx_1, row in tqdm(input_df.iterrows(), total=total_rows, desc="Processing rows"):
            hook_text = row['Hook Text']
            hook_number = idx_1 + 1

            process_audios(ELEVENLABS_API_KEY, row, hook_number, hook_text, input_df, idx_1, output_audios_folder, INPUT_FILE, voice_id)

        current_thread_count = 0
        for idx, row in tqdm(input_df.iterrows(), total=total_rows, desc="Processing rows"):
            hook_text = row['Hook Text']
            hook_number = idx + 1

            if row['Hook Video Filename'] != '' and os.path.exists(os.path.join(output_videos_folder, row['Hook Video Filename'])):
                continue

            audio_clip = AudioFileClip(os.path.join(output_audios_folder, row['Audio Filename']))
            video_index = idx % len(video_files)
            num_videos_to_use = int(round(audio_clip.duration / 2))

            video_file_size = len(video_files)
            if num_videos_to_use + video_index > video_file_size:
                num_videos_to_use = video_file_size - video_index

            last_video = video_index + num_videos_to_use
            video_files_to_use = [os.path.join(input_videos_folder, video_files[i]) for i in range(video_index, last_video)]

            if params['task_id'] in canceled_tasks:
                return handle_task_cancellation(temp_dir, task_id)

            hook_job = threading.Thread(target=process_audio_on_videos, args=(row, video_files_to_use, idx, input_df, hook_number, hook_text, num_videos_to_use, audio_clip, OUT_VIDEO_WIDTH, OUT_VIDEO_HEIGHT, output_videos_folder, INPUT_FILE, total_rows, task_id))
            hook_job.start()
            all_hooks.append(hook_job)
            current_thread_count += 1
            if current_thread_count == int(no_of_parallel_executions):
                for hook in all_hooks:
                    hook.join()
                all_hooks.clear()
                current_thread_count = 0

        for hook in all_hooks:
            hook.join()

        script_files = sorted([os.path.join(input_scripts_folder, f) for f in os.listdir(input_scripts_folder) if f.endswith('.mp4')])
        hook_files = sorted([os.path.join(output_videos_folder, f) for f in os.listdir(output_videos_folder) if f.endswith('.mp4')])

        current_thread_count = 0
        for idx, script in enumerate(script_files):
            for idy, hook in enumerate(tqdm(hook_files, desc=f"Processing Script {idx + 1} hooks")):
                if params['task_id'] in canceled_tasks:
                    return handle_task_cancellation(temp_dir, task_id)

                hook_script_filename = f"{os.path.splitext(os.path.basename(hook))[0]}_{os.path.splitext(os.path.basename(script))[0]}.mp4".replace(" ", "_")
                temp_filename = os.path.join(SCRIPTS_OUTPUT_DIR, f"temp_{idx}_{idy}.mp4")
                final_filename = os.path.join(SCRIPTS_OUTPUT_DIR, hook_script_filename)
                if os.path.isfile(final_filename):
                    continue

                hook_job = threading.Thread(target=process_script_file, args=(hook, script, temp_filename, final_filename, idy, idx, output_videos_folder, OUT_VIDEO_WIDTH, OUT_VIDEO_HEIGHT, total_rows, task_id))
                hook_job.start()
                all_hooks.append(hook_job)
                current_thread_count += 1
                if current_thread_count == int(no_of_parallel_executions):
                    for hook in all_hooks:
                        hook.join()
                    all_hooks.clear()
                    current_thread_count = 0

        for hook in all_hooks:
            hook.join()

        calculate_total_hours(start_time)
        zip_output_folder(SCRIPTS_OUTPUT_DIR)

        # Emit task completion
        socketio.emit('task_complete', {'task_id': task_id})

        zip_path = f"{SCRIPTS_OUTPUT_DIR}.zip"
        print(f"Debug: Path to the zip file: {zip_path}")
        if os.path.exists(zip_path):
            print("Debug: Zip file exists.")
            print("Debug: Contents of the zip file:")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.printdir()
        else:
            print("Debug: Zip file does not exist!")
            
    except Exception as e:
        # Emit error message
        socketio.emit('error', {'task_id': task_id, 'message': str(e)})
        print(f"Error during processing: {e}")
        delete_temp_dir(temp_dir)

@app.route('/download_output', methods=['GET'])
def download_output():
    if os.path.exists(OUTPUT_ZIP_PATH):
        return send_file(OUTPUT_ZIP_PATH, as_attachment=True)
    else:
        return jsonify({"error": "File not found"}), 404

def zip_output_folder(folder_path):
    print(f"Debug: Zipping the folder {folder_path}...")
    shutil.make_archive(folder_path, 'zip', folder_path)
    print(f"Debug: Zipped the folder {folder_path} successfully.")
    print(f"Debug: Checking if zip file exists at {folder_path}.zip")
    if os.path.exists(f"{folder_path}.zip"):
        print("Debug: Zip file found.")
    else:
        print("Debug: Zip file not found!")

def process_audios(api_key, row, hook_number, hook_text, input_df, idx, output_audios_folder, INPUT_FILE, voice_id):
    if row['Audio Filename'] in (None, '') or not os.path.exists(os.path.join(output_audios_folder, row['Audio Filename'])):
        print(f"Generating voiceover for hook {hook_number}...")
        audio_filename = os.path.join(output_audios_folder, f'hook_{hook_number}.mp3')
        status, voice_name = text_to_speech_file(api_key, hook_text, audio_filename, voice_id)
        row['Voice'] = voice_name
        row['Audio Filename'] = os.path.basename(audio_filename)
        input_df.at[idx, 'Voice'] = voice_name
        input_df.at[idx, 'Audio Filename'] = row['Audio Filename']
        input_df.to_csv(INPUT_FILE, index=False, quotechar='"', quoting=csv.QUOTE_ALL, escapechar='\\')

def text_to_speech_file(api_key, text: str, save_file_path: str, voice_id: str, remove_punctuation: bool = True) -> bool:
    if remove_punctuation:
        text = text.replace('-', ' ').replace('"', ' ').replace("'", ' ')
        text = re.sub(r'[^\w\s]', '', text)

    url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
    headers = {
        "Accept": "audio/mpeg",
        "Content-Type": "application/json",
        "xi-api-key": api_key
    }
    data = {
        "text": text,
        "model_id": "eleven_monolingual_v1",
        "voice_settings": {
            "stability": 0.5,
            "similarity_boost": 0.75
        }
    }

    response = requests.post(url, json=data, headers=headers)

    if response.status_code != 200:
        raise Exception(f"API request failed with status code {response.status_code}: {response.text}")

    with open(save_file_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=1024):
            if chunk:
                f.write(chunk)

    return True, voice_id

def process_audio_on_videos(row, video_files, idx, input_df, hook_number, hook_text, num_videos_to_use, audio_clip, OUT_VIDEO_WIDTH, OUT_VIDEO_HEIGHT, output_videos_folder, INPUT_FILE, total_rows, task_id):
    row['Input Video Filename'] = [os.path.basename(considered_video) for considered_video in video_files]
    input_df.at[idx, 'Input Video Filename'] = row['Input Video Filename']
    input_df.to_csv(INPUT_FILE, index=False, quotechar='"', quoting=csv.QUOTE_ALL)

    each_video_duration = audio_clip.duration / num_videos_to_use
    video_clips = []
    for considered_vid in video_files:
        video_clip = VideoFileClip(considered_vid).resize(width=OUT_VIDEO_WIDTH)
        (w, h) = video_clip.size
        cropped_clip = crop(video_clip.subclip(0, each_video_duration), width=OUT_VIDEO_WIDTH, height=OUT_VIDEO_HEIGHT, x_center=w / 2, y_center=h / 2)
        video_clips.append(cropped_clip)

    final_video_clip = concatenate_videoclips(video_clips)
    custom_text_clip = create_custom_text_clip(hook_text, OUT_VIDEO_WIDTH, OUT_VIDEO_HEIGHT)

    final_clip = CompositeVideoClip([
        final_video_clip.audio_fadein(0.2).audio_fadeout(0.2),
        final_video_clip,
        custom_text_clip
    ]).set_audio(audio_clip).set_duration(audio_clip.duration)

    output_video_filename = os.path.join(output_videos_folder, f'hook_{idx}.mp4')

    final_clip.write_videofile(output_video_filename, temp_audiofile=os.path.join(output_videos_folder, f"temp-audio_{idx}.m4a"), remove_temp=True, codec='libx264', audio_codec="aac")

    input_df.at[idx, 'Hook Video Filename'] = os.path.basename(output_video_filename)
    input_df.to_csv(INPUT_FILE, index=False, quotechar='"', quoting=csv.QUOTE_ALL, escapechar='\\')

def create_custom_text_clip(hook_text, OUT_VIDEO_WIDTH, OUT_VIDEO_HEIGHT):
    hook_text = split_hook_text(hook_text)
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

    return final_clip

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

def calculate_total_hours(start_time):
    end_time = datetime.now()
    time_difference = end_time - start_time
    total_seconds = time_difference.total_seconds()
    total_hours = total_seconds / 3600
    print(f'total_hours={total_hours}')

def process_script_file(hook, script, temp_filename, final_filename, idy, idx, output_videos_folder, OUT_VIDEO_WIDTH, OUT_VIDEO_HEIGHT, total_rows, task_id):
    hook_clip = VideoFileClip(hook)
    script_clip = VideoFileClip(script).resize(width=OUT_VIDEO_WIDTH)
    (w, h) = script_clip.size
    script_clip = crop(script_clip, width=OUT_VIDEO_WIDTH, height=OUT_VIDEO_HEIGHT, x_center=w / 2, y_center=h / 2)

    final_clip = concatenate_videoclips([hook_clip, script_clip])
    final_clip.write_videofile(temp_filename, temp_audiofile=os.path.join(output_videos_folder, f"temp-audio_{idx}_{idy}.m4a"), remove_temp=True, codec='libx264', audio_codec="aac")

    shutil.move(temp_filename, final_filename)

    # Emit progress after moving output file to the final destination
    socketio.emit('progress', {'task_id': task_id, 'progress': ((idx + 1) / total_rows) * 100, 'step': f"{idx + 1}/{total_rows}"})
    print(f"Moved {temp_filename} to {final_filename}")

@app.route('/cancel_task', methods=['POST'])
def cancel_task():
    try:
        task_id = request.json.get('task_id')
        print(f"Received request to cancel task: {task_id}")
        with tasks_lock:
            if task_id in canceled_tasks:
                print(f"Task {task_id} already marked for cancellation")
            canceled_tasks.add(task_id)
            if task_id in upload_tasks:
                upload_tasks[task_id] = False
                for pid in process_pids.get(task_id, []):
                    try:
                        os.kill(pid, signal.SIGTERM)
                    except ProcessLookupError:
                        pass
                process_pids.pop(task_id, None)
                print(f"Task {task_id} set to be canceled")
        return jsonify({"message": "Task cancellation request processed"}), 200
    except Exception as e:
        print(f"Error handling cancel task request: {e}")
        return jsonify({"error": "Internal server error"}), 500

def handle_task_cancellation(temp_dir, task_id):
    delete_temp_dir(temp_dir)
    socketio.emit('task_cancelled', {'task_id': task_id})

if __name__ == '__main__':
    socketio.run(app, debug=True, host='0.0.0.0')
