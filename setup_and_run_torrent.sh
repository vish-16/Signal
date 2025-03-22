#!/bin/bash

# Exit on any error
set -e

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}Starting setup for Torrent Downloader in GitHub Workspace...${NC}"

# Step 1: Install dependencies
echo "Updating package lists and installing python3-libtorrent..."
sudo apt-get update -y
sudo apt-get install -y python3-libtorrent
echo -e "${GREEN}Dependencies installed.${NC}"

# Step 2: Create the Python script
echo "Creating torrent_downloader.py..."
cat << 'EOF' > torrent_downloader.py
import libtorrent as lt
import time
import os
import zipfile
from threading import Thread
import shutil
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

class TorrentDownloader:
    def __init__(self, torrent_source, save_path="./downloads"):
        print("Initializing torrent downloader...")
        self.ses = lt.session()
        self.ses.listen_on(6881, 6891)
        
        settings = self.ses.get_settings()
        settings['download_rate_limit'] = 0
        settings['upload_rate_limit'] = 0
        settings['active_downloads'] = 8
        settings['active_seeds'] = 8
        settings['connections_limit'] = 200
        settings['max_peerlist_size'] = 1000
        self.ses.apply_settings(settings)
        
        self.save_path = save_path
        if not os.path.exists(save_path):
            os.makedirs(save_path)
            
        params = {
            'save_path': save_path,
            'storage_mode': lt.storage_mode_t.storage_mode_sparse
        }
        
        if torrent_source.startswith('magnet:'):
            self.handle = lt.add_magnet_uri(self.ses, torrent_source, params)
        else:
            torrent_info = lt.torrent_info(torrent_source)
            self.handle = self.ses.add_torrent({'ti': torrent_info, 'save_path': save_path})
        print("Torrent handle created.")

    def get_progress(self):
        s = self.handle.status()
        return s.progress * 100

    def get_speed(self):
        s = self.handle.status()
        return s.download_rate / 1024 / 1024

    def get_peers(self):
        s = self.handle.status()
        return s.num_peers

    def is_finished(self):
        s = self.handle.status()
        return s.is_seeding or s.progress == 1.0

    def download(self):
        print("🚀 Initiating Download Sequence...")
        print(f"📍 Target Zone: {self.save_path}")
        
        bar_length = 30
        stages = ["🌑", "🌒", "🌓", "🌔", "🌕"]
        
        while not self.is_finished():
            s = self.handle.status()
            progress = self.get_progress()
            speed = self.get_speed()
            peers = self.get_peers()
            
            filled = int(bar_length * progress / 100)
            stage_idx = min(int(progress / 20), len(stages) - 1)
            bar = stages[stage_idx] * filled + "⬜" * (bar_length - filled)
            
            print(f'\r⚡ [{bar}] {progress:.1f}% '
                  f'| 🚄 {speed:.2f} MB/s '
                  f'| 👥 {peers} '
                  f'| 🔄 {s.state}', end='')
            
            self.handle.force_reannounce()
            time.sleep(1)
        
        print("\n🎉 Download Mission Accomplished!")
        
        file_info = self.handle.torrent_file()
        for file in file_info.files():
            print(f"📦 {file.path} ({file.size / 1024 / 1024:.2f} MB)")

def create_zip(source_path, zip_name="downloaded_content.zip"):
    zip_path = os.path.join(os.getcwd(), zip_name)
    print(f"🗜️ Compressing main folder into {zip_name}...")
    
    total_size = 0
    main_folder = None
    for root, dirs, files in os.walk(source_path):
        if dirs and not main_folder:
            main_folder = os.path.join(root, dirs[0])
        for file in files:
            total_size += os.path.getsize(os.path.join(root, file))
    
    if not main_folder:
        main_folder = source_path
    
    bar_length = 30
    compressed_size = 0
    
    with zipfile.ZipFile(zip_path, 'w', compression=zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(main_folder):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, source_path)
                zipf.write(file_path, arcname)
                compressed_size += os.path.getsize(file_path)
                
                progress = (compressed_size / total_size) * 100 if total_size > 0 else 100
                filled = int(bar_length * progress / 100)
                bar = "🟥" * filled + "⬜" * (bar_length - filled)
                print(f'\r📦 [{bar}] {progress:.1f}% - Adding: {file}', end='')
    
    print(f"\n✅ Compression completed: {zip_name}")
    return zip_path

def save_to_workspace(source_path, output_dir="./output"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    files_list = [f for f in os.listdir(source_path) if os.path.isfile(os.path.join(source_path, f))]
    if len(files_list) == 1 and files_list[0].endswith('.zip'):
        zip_file = os.path.join(source_path, files_list[0])
        print(f"📦 Detected existing zip: {files_list[0]}")
    else:
        zip_file = create_zip(source_path)
    
    dest_file = os.path.join(output_dir, os.path.basename(zip_file))
    print("\n📂 Saving zip to workspace...")
    shutil.move(zip_file, dest_file)
    print(f"📤 Saved: {os.path.basename(dest_file)}")

def main():
    print("Starting main execution...")
    
    choice = input("Enter '1' for magnet link or '2' to upload .torrent file path: ")
    print(f"User chose: {choice}")
    
    if choice == '1':
        torrent_source = input("Enter magnet link: ")
        print(f"Magnet link provided: {torrent_source}")
    elif choice == '2':
        torrent_source = input("Enter the full path to your .torrent file: ")
        if not os.path.exists(torrent_source):
            print(f"File not found: {torrent_source}. Exiting...")
            return
        print(f"Torrent file path provided: {torrent_source}")
    else:
        print("Invalid choice. Exiting...")
        return
    
    downloader = TorrentDownloader(torrent_source)
    
    download_thread = Thread(target=downloader.download)
    download_thread.start()
    download_thread.join()
    
    save_to_workspace(downloader.save_path)
    print("Script execution completed.")

if __name__ == "__main__":
    main()
EOF
echo -e "${GREEN}torrent_downloader.py created.${NC}"

# Step 3: Make the script executable (optional, not needed for Python but good practice)
chmod +x torrent_downloader.py

# Step 4: Interactive menu
echo -e "${GREEN}Torrent Downloader Setup Complete!${NC}"
echo "Choose an option to proceed:"
echo "1) Download via Magnet Link"
echo "2) Download via .torrent File Path"
echo "3) Exit"

read -p "Enter your choice (1-3): " choice

case $choice in
    1)
        read -p "Enter magnet link: " magnet_link
        if [[ "$magnet_link" =~ ^magnet: ]]; then
            echo -e "${GREEN}Starting download with magnet link...${NC}"
            python3 torrent_downloader.py <<< "1"$'\n'"$magnet_link"
        else
            echo -e "${RED}Invalid magnet link format. Exiting...${NC}"
            exit 1
        fi
        ;;
    2)
        read -p "Enter the full path to your .torrent file: " torrent_path
        if [ -f "$torrent_path" ]; then
            echo -e "${GREEN}Starting download with torrent file...${NC}"
            python3 torrent_downloader.py <<< "2"$'\n'"$torrent_path"
        else
            echo -e "${RED}File not found: $torrent_path. Exiting...${NC}"
            exit 1
        fi
        ;;
    3)
        echo -e "${GREEN}Exiting...${NC}"
        exit 0
        ;;
    *)
        echo -e "${RED}Invalid choice. Exiting...${NC}"
        exit 1
        ;;
esac
