🎧 Audio Similarity App — Setup & Usage Guide (macOS)

This guide walks you through, step-by-step, how to install and run the Audio Similarity App on a Mac — even if you’ve never used Python before.

🧰 1. Install Homebrew (package manager for Mac)

📝 Homebrew makes it easy to install the software you’ll need, like Python and FFmpeg.

Open Terminal (press Cmd + Space, type “Terminal”, hit Enter)

Copy and paste this line and press Enter:

/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"


Wait until it finishes. Then run:

brew doctor


If it says “Your system is ready to brew”, you’re good to go.

🐍 2. Install Python 3

You need Python 3.10 or newer.

brew install python


Check it’s installed:

python3 --version


You should see something like Python 3.11.x.

🎧 3. Install FFmpeg (for audio processing)

The app uses FFmpeg to read and segment audio files.

brew install ffmpeg

📦 4. Download the project from GitHub

In Terminal, navigate to where you want to keep the app (for example, your Desktop):

cd ~/Desktop


Then clone the GitHub repository (replace the URL if yours is different):

git clone https://github.com/poynta/audio-similarity-app.git


Now go into that folder:

cd audio-similarity-app

🧙‍♂️ 5. Create a virtual environment (so your dependencies don’t mess with system Python)

python3 -m venv .venv


Activate it:

source .venv/bin/activate


Your terminal should now show something like:

(.venv) macbook@MacBook-Air audio-similarity-app %

📚 6. Install all required Python libraries

Run this:

pip install -r requirements.txt


This will install:

FreeSimpleGUI (for the interface)

laion-clap (for audio embeddings)

faiss (for similarity search)

numpy and others

🕐 It might take a few minutes, especially the first time.

🚀 7. Run the app

In the same Terminal (while .venv is active):

python3 audio_gui6.py


After a few seconds, a window will open titled:

Audio Similarity (CLAP + FAISS) — Searchable Index

🎛️ 8. Using the App

Index Audio Folder

Click “Browse” and select a folder containing a few audio files (start small to test).

Then click “Index”.

Wait for it to finish

Search Indexed Track

Start typing a file name from your indexed folder — matches will appear.

Click “Query Indexed” to find similar tracks.

Query External File

Use “Browse” to pick any audio file on your computer (even if not indexed).

Enter a number (like 5) to choose how many similar tracks to return.

Click “Query”.

Right-click results

Right-click on any result to Query This Track or Show in Folder.

Clear Index

Click “Clear Index” to delete your current index.

(⚠️ A popup will ask for confirmation before deleting anything.)

🧹 9. To stop the app

In Terminal, press:

Ctrl + C


To close the virtual environment:

deactivate

💡 10. For next time (quick start)

If you’ve already installed everything once, next time you just need to:

cd ~/Desktop/audio-similarity-app
source .venv/bin/activate
python3 audio_gui6.py

🪄 Optional: Update the app from GitHub

If you make changes or I update the repo, you can pull the latest version:

git pull

🧱 Troubleshooting
Problem	Fix
“ffmpeg not found”	Run brew install ffmpeg again
“faiss not found”	Try pip install faiss-cpu
“No module named laion_clap”	Run pip install laion-clap
Window doesn’t open	Make sure you’re using Python 3.10+ and .venv is activated
Model takes long to load	That’s normal — CLAP is a big neural model (loads once per session)


















RUN THE WEB APP (NOT WORKING)

1️⃣ Open Terminal / Command Prompt

macOS: Open Terminal.

Windows: Open Command Prompt or PowerShell.

2️⃣ Download the GitHub repo


Run this command
--------------------
git clone https://github.com/poynta/audio_similarity-app.git
cd audio_sim
--------------------
This downloads the project to your computer and moves into the folder.



3️⃣ Install Python and dependencies


macOS:
--------------------
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
--------------------


Windows (PowerShell):
--------------------
python -m venv .venv
.venv\Scripts\Activate.ps1   # or use .venv\Scripts\activate.bat for cmd
pip install -r requirements.txt
--------------------

⚠️ Make sure Python 3.10+ is installed.



4️⃣ Run the app

Start the app with:
--------------------
python audio_web.py
--------------------

You should see something like:

* Running on local URL:  http://127.0.0.1:7860



5️⃣ Open the app in your browser


Open a web browser and go to:
--------------------
http://127.0.0.1:7860
--------------------
This is your local app interface.



6️⃣ Index your audio

Go to the “Index Folder” tab.

Select a folder of audio files (start small for testing).

Click Start Indexing.

Wait for it to finish — you’ll see a progress update.



7️⃣ Search / Query audio

Switch to the “Query / Upload” tab.

Either:

Upload a file to search for similar tracks in the index, or

Type a track name from the index to query it.

View the results directly in the table.



💡 Tip: Start with a small folder of 5–10 tracks to make sure everything works before indexing a large library.
