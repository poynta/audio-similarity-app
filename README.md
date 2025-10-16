Audio Similarity App — Quick Start
1️⃣ Open Terminal

On macOS, open the Terminal app.

2️⃣ Download the GitHub repo

Run this command (replace the URL with your repo’s actual GitHub URL):

git clone https://github.com/yourusername/audio_similarity.git
cd audio_similarity


This downloads the project to your computer and moves into the folder.

3️⃣ Install requirements

If you haven’t already, create a virtual environment (optional, but recommended) and install dependencies:

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

4️⃣ Run the app

Start the app with:

python audio_web_tabs.py


You should see something like:

* Running on local URL:  http://127.0.0.1:7860

5️⃣ Open the app in your browser

Open a web browser and go to:

http://127.0.0.1:7860


This is your local app interface.

6️⃣ Index your audio

Go to the “Index Folder” tab.

Select a folder of audio files (start small for testing).

Click Start Indexing.

Wait for it to finish — you’ll see a progress update.

7️⃣ Search/query audio

Switch to the “Query / Upload” tab.

Either:

Upload a file to search for similar tracks in the index, or

Type a track name from the index to query it.

View the results directly in the table.
