import sys
import os
import io
import threading
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QLabel, QLineEdit, QPushButton, QTextEdit, QFileDialog, QGroupBox)
from PyQt6.QtGui import QFont, QColor
from PyQt6.QtCore import Qt, QObject, pyqtSignal

# Import the extraction logic
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from extract_vocals import extract_vocals, setup_model_dir

class LogSignal(QObject):
    text_written = pyqtSignal(str)

    def write(self, text):
        self.text_written.emit(str(text))
    
    def flush(self):
        pass

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SAM Audio Extractor")
        self.resize(600, 650)
        
        # --- Layout ---
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setSpacing(15)
        main_layout.setContentsMargins(20, 20, 20, 20)

        # Header
        header_label = QLabel("SAM Audio Separator")
        header_label.setFont(QFont("Arial", 16, QFont.Weight.Bold))
        header_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        main_layout.addWidget(header_label)

        # File Selection
        file_group = QGroupBox("Input Audio")
        file_layout = QHBoxLayout()
        self.file_entry = QLineEdit()
        browse_btn = QPushButton("Browse...")
        browse_btn.clicked.connect(self.browse_file)
        file_layout.addWidget(self.file_entry)
        file_layout.addWidget(browse_btn)
        file_group.setLayout(file_layout)
        main_layout.addWidget(file_group)

        # Filters
        filter_group = QGroupBox("Separation Filter")
        filter_layout = QVBoxLayout()
        
        filter_layout.addWidget(QLabel("Enter prompt (what to extract):"))
        self.prompt_entry = QLineEdit("vocals")
        filter_layout.addWidget(self.prompt_entry)

        # Quick Buttons
        btn_layout = QHBoxLayout()
        for label in ["Vocals", "Music", "Drums", "Bass"]:
            btn = QPushButton(label)
            btn.clicked.connect(lambda checked, t=label: self.set_prompt(t))
            btn_layout.addWidget(btn)
        filter_layout.addLayout(btn_layout)
        filter_group.setLayout(filter_layout)
        main_layout.addWidget(filter_group)

        # Run Button
        self.run_btn = QPushButton("Extract Audio")
        self.run_btn.setFixedHeight(40)
        self.run_btn.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold; font-size: 14px;")
        self.run_btn.clicked.connect(self.run_extraction)
        main_layout.addWidget(self.run_btn)

        # Console Output
        log_group = QGroupBox("Log Output")
        log_layout = QVBoxLayout()
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setStyleSheet("background-color: #2b2b2b; color: #dcdcdc; font-family: Consolas;")
        log_layout.addWidget(self.log_text)
        log_group.setLayout(log_layout)
        main_layout.addWidget(log_group)

        # Redirect Stdout
        self.log_signal = LogSignal()
        self.log_signal.text_written.connect(self.append_log)
        sys.stdout = self.log_signal
        sys.stderr = self.log_signal

        # Init Setup in Background
        threading.Thread(target=self.init_setup, daemon=True).start()

    def append_log(self, text):
        cursor = self.log_text.textCursor()
        cursor.movePosition(cursor.MoveOperation.End)
        cursor.insertText(text)
        self.log_text.setTextCursor(cursor)
        self.log_text.ensureCursorVisible()

    def init_setup(self):
        print("Checking model files...")
        try:
            setup_model_dir()
            print("Ready.")
        except Exception as e:
            print(f"Error checking model: {e}")

    def browse_file(self):
        filename, _ = QFileDialog.getOpenFileName(self, "Select Audio File", "", "Audio Files (*.wav *.mp3 *.flac *.ogg);;All Files (*)")
        if filename:
            self.file_entry.setText(filename)

    def set_prompt(self, text):
        self.prompt_entry.setText(text.lower())

    def run_extraction(self):
        audio_path = self.file_entry.text().strip()
        prompt = self.prompt_entry.text().strip()
        
        if not audio_path or not os.path.exists(audio_path):
            print("Error: Please select a valid audio file.")
            return
        if not prompt:
            print("Error: Please enter a prompt.")
            return

        self.run_btn.setEnabled(False)
        self.run_btn.setText("Processing...")
        
        threading.Thread(target=self._process_thread, args=(audio_path, prompt), daemon=True).start()

    def _process_thread(self, audio_path, prompt):
        try:
            print(f"Starting extraction for '{prompt}'...")
            extract_vocals(audio_path, output_dir=os.path.dirname(audio_path), prompts=[prompt])
            print("Extraction complete!")
        except Exception as e:
            import traceback
            traceback.print_exc() 
            # Traceback will be printed to stderr, which is redirected to log
            print(f"Extraction failed: {e}")
        finally:
            # Update UI via main thread (sort of safe with signal, but button enablement should be safe here or via signal)
            # PyQt requires UI updates on main thread. 
            # We can use QMetaObject.invokeMethod but for simplicity let's use a signal or basic logic.
            # Actually, accessing widgets from thread is unsafe. 
            # Let's simple-hack it or do it properly. 
            # Proper way: signal. 
            pass
            # I will just re-enable it on next click? No.
            # I need a signal to finish.
            # Due to complexity of one-shot file rewrite, I'll skip the signal for button reset for now 
            # or add a simple timer check? 
            # No, let's add the signal properly.
            
            # Since I can't edit the class easily inside this string, I'll rely on the user restarting app for now 
            # OR I add a signal to the class now.
            
    # Correction: I am writing the file now so I can add the signal.

class MainWindowWithSignals(MainWindow):
     # I can't inherit if I am overwriting the file. 
     # I will edit the MainWindow class in the string above.
     pass

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    
    # Quick patch to enable button reset from thread
    from PyQt6.QtCore import pyqtSignal
    class WorkerSignals(QObject):
        finished = pyqtSignal()
    
    signals = WorkerSignals()
    signals.finished.connect(lambda: [window.run_btn.setEnabled(True), window.run_btn.setText("Extract Audio")])
    
    # Monkey patch the process thread to emit signal
    original_process = window._process_thread
    def patched_process(audio_path, prompt):
        original_process(audio_path, prompt)
        signals.finished.emit()
    
    window._process_thread = patched_process
    
    window.show()
    sys.exit(app.exec())
