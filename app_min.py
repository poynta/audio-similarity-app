#!/usr/bin/env python3
"""
app_min.py
Minimal Tkinter launcher to validate GUI startup and packaging.
- Shows a lightweight window immediately
- Optionally launches the full FreeSimpleGUI app (audio_gui5) lazily

Intended for PyInstaller windowed builds as a smoke test for launch reliability.
"""

import sys
import threading
import tkinter as tk
from tkinter import ttk, messagebox


def launch_full_app(root: tk.Tk):
    """Destroy the stub window and lazily import the heavy GUI."""
    try:
        # Close the minimal window before importing heavy modules
        root.destroy()
        # Lazy import so the initial window appears instantly
        import audio_gui5  # noqa: F401 - side-effect opens the main app window
    except Exception as e:
        messagebox.showerror("Launch Error", f"Failed to launch full app.\n{e}")


def main():
    root = tk.Tk()
    root.title("Audio Similarity — Launcher")
    root.geometry("420x200")

    container = ttk.Frame(root, padding=16)
    container.pack(fill=tk.BOTH, expand=True)

    title = ttk.Label(container, text="Audio Similarity", font=("Helvetica", 18, "bold"))
    title.pack(anchor=tk.W, pady=(0, 8))

    desc = ttk.Label(
        container,
        text=(
            "This minimal launcher verifies the GUI can start.\n"
            "Click 'Launch Full App' to load the full experience."
        ),
        justify=tk.LEFT,
        wraplength=380,
    )
    desc.pack(anchor=tk.W, pady=(0, 12))

    progress = ttk.Progressbar(container, mode="indeterminate")
    progress.pack(fill=tk.X, pady=(0, 12))
    progress.start(14)  # gentle idle animation

    btn_row = ttk.Frame(container)
    btn_row.pack(fill=tk.X)

    launch_btn = ttk.Button(
        btn_row,
        text="Launch Full App",
        command=lambda: threading.Thread(target=launch_full_app, args=(root,), daemon=True).start(),
    )
    launch_btn.pack(side=tk.LEFT)

    quit_btn = ttk.Button(btn_row, text="Quit", command=root.destroy)
    quit_btn.pack(side=tk.RIGHT)

    root.mainloop()


if __name__ == "__main__":
    main()
