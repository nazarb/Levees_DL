#!/usr/bin/env python3
"""
MEGA Archaeological Data Downloader
Download and manage satellite imagery files from MEGA for archaeological analysis.
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Optional
import subprocess


class MegaDownloader:
    """Handle MEGA file downloads for archaeological satellite imagery."""
    
    # Configuration for sectors and their associated files
    SECTORS = {
        "CFE_a": {
            "url": "https://mega.nz/file/E9gBCCjR#UCDklOgOVOwQ0hRAIy5pvAdX9Y-VLBQktDlsrj4R1Ms",
            "files": [
                "CFE_a_selected_L5_S2_S1_MSRM_PCA_GLO_N48.tif",
                "CFE_a_selected_L5_S2_S1_MSRM_PCA.tif",
                "CFE_a_selected_L5_S2_S1.tif",
            ]
        },
        "CFE_c": {
            "url": "https://mega.nz/file/PLACEHOLDER#PLACEHOLDER",  # Add actual URL
            "files": [
                "CFE_c_selected_L5_S2_S1_MSRM_PCA_GLO_N48.tif",
                "CFE_c_selected_L5_S2_S1_MSRM_PCA.tif",
                "CFE_c_selected_L5_S2_S1.tif",
            ]
        }
    }
    
    def __init__(self, output_dir: str = "./data"):
        """
        Initialize the downloader.
        
        Args:
            output_dir: Directory to save downloaded files
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def check_megacmd(self) -> bool:
        """Check if MEGAcmd is installed."""
        try:
            result = subprocess.run(
                ["mega-version"],
                capture_output=True,
                text=True
            )
            return result.returncode == 0
        except FileNotFoundError:
            return False
    
    def install_instructions(self):
        """Print installation instructions for MEGAcmd."""
        print("\n" + "="*60)
        print("MEGAcmd is not installed!")
        print("="*60)
        print("\nInstallation instructions:")
        print("\nUbuntu/Debian:")
        print("  wget https://mega.nz/linux/repo/xUbuntu_22.04/amd64/megacmd-xUbuntu_22.04_amd64.deb")
        print("  sudo apt install ./megacmd-xUbuntu_22.04_amd64.deb")
        print("\nAlternatively, install via pip:")
        print("  pip install mega.py")
        print("\nOr use megatools:")
        print("  sudo apt install megatools")
        print("="*60 + "\n")
    
    def display_sectors(self):
        """Display available sectors."""
        print("\n" + "="*60)
        print("Available Sectors:")
        print("="*60)
        for i, (sector, info) in enumerate(self.SECTORS.items(), 1):
            print(f"{i}. {sector}")
            print(f"   URL: {info['url'][:50]}...")
            print(f"   Files: {len(info['files'])}")
        print("="*60 + "\n")
    
    def display_files(self, sector: str):
        """Display available files for a sector."""
        if sector not in self.SECTORS:
            print(f"Error: Sector '{sector}' not found!")
            return
        
        files = self.SECTORS[sector]["files"]
        print("\n" + "="*60)
        print(f"Available Files for {sector}:")
        print("="*60)
        for i, file in enumerate(files, 1):
            print(f"{i}. {file}")
        print("="*60 + "\n")
    
    def select_sector(self) -> Optional[str]:
        """Interactive sector selection."""
        self.display_sectors()
        
        while True:
            try:
                choice = input("Select sector (enter number or name, or 'q' to quit): ").strip()
                
                if choice.lower() == 'q':
                    return None
                
                # Check if it's a number
                if choice.isdigit():
                    idx = int(choice) - 1
                    sectors = list(self.SECTORS.keys())
                    if 0 <= idx < len(sectors):
                        return sectors[idx]
                
                # Check if it's a sector name
                if choice in self.SECTORS:
                    return choice
                
                print(f"Invalid selection: '{choice}'. Please try again.")
                
            except KeyboardInterrupt:
                print("\nCancelled.")
                return None
    
    def select_file(self, sector: str) -> Optional[str]:
        """Interactive file selection for a sector."""
        self.display_files(sector)
        files = self.SECTORS[sector]["files"]
        
        while True:
            try:
                choice = input("Select file (enter number or name, or 'q' to quit): ").strip()
                
                if choice.lower() == 'q':
                    return None
                
                # Check if it's a number
                if choice.isdigit():
                    idx = int(choice) - 1
                    if 0 <= idx < len(files):
                        return files[idx]
                
                # Check if it's a filename
                if choice in files:
                    return choice
                
                print(f"Invalid selection: '{choice}'. Please try again.")
                
            except KeyboardInterrupt:
                print("\nCancelled.")
                return None
    
    def download_file(self, sector: str, filename: str) -> bool:
        """
        Download a file from MEGA.
        
        Args:
            sector: Sector name
            filename: File to download
            
        Returns:
            True if download successful, False otherwise
        """
        if sector not in self.SECTORS:
            print(f"Error: Sector '{sector}' not found!")
            return False
        
        url = self.SECTORS[sector]["url"]
        output_path = self.output_dir / sector / filename
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        print(f"\nDownloading: {filename}")
        print(f"Sector: {sector}")
        print(f"Output: {output_path}")
        print("-" * 60)
        
        # Try MEGAcmd first
        if self.check_megacmd():
            return self._download_with_megacmd(url, output_path)
        
        # Fallback to mega.py
        return self._download_with_megapy(url, output_path)
    
    def _download_with_megacmd(self, url: str, output_path: Path) -> bool:
        """Download using MEGAcmd."""
        try:
            cmd = [
                "mega-get",
                url,
                str(output_path)
            ]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                print(f"✓ Download complete: {output_path}")
                return True
            else:
                print(f"✗ Download failed: {result.stderr}")
                return False
                
        except Exception as e:
            print(f"✗ Error during download: {e}")
            return False
    
    def _download_with_megapy(self, url: str, output_path: Path) -> bool:
        """Download using mega.py library."""
        try:
            from mega import Mega
            
            print("Using mega.py for download...")
            mega = Mega()
            m = mega.login()
            
            # Download file
            m.download_url(url, dest_path=str(output_path.parent), dest_filename=output_path.name)
            
            print(f"✓ Download complete: {output_path}")
            return True
            
        except ImportError:
            print("✗ mega.py not installed. Install with: pip install mega.py")
            self.install_instructions()
            return False
        except Exception as e:
            print(f"✗ Error during download: {e}")
            return False
    
    def run_interactive(self):
        """Run interactive mode."""
        print("\n" + "="*60)
        print("MEGA Archaeological Data Downloader")
        print("="*60)
        
        # Select sector
        sector = self.select_sector()
        if not sector:
            print("Cancelled.")
            return
        
        # Select file
        filename = self.select_file(sector)
        if not filename:
            print("Cancelled.")
            return
        
        # Download
        success = self.download_file(sector, filename)
        
        if success:
            print("\n" + "="*60)
            print("Download Complete!")
            print("="*60)
            output_path = self.output_dir / sector / filename
            print(f"File saved to: {output_path}")
            print(f"File size: {output_path.stat().st_size / (1024**2):.2f} MB")
        else:
            print("\n" + "="*60)
            print("Download Failed!")
            print("="*60)


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Download archaeological satellite imagery from MEGA",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode
  python mega_downloader.py
  
  # Direct download
  python mega_downloader.py --sector CFE_a --file CFE_a_selected_L5_S2_S1_MSRM_PCA_GLO_N48.tif
  
  # Specify output directory
  python mega_downloader.py --output-dir ./my_data
        """
    )
    
    parser.add_argument(
        "--sector",
        choices=list(MegaDownloader.SECTORS.keys()),
        help="Sector to download from"
    )
    
    parser.add_argument(
        "--file",
        help="File to download"
    )
    
    parser.add_argument(
        "--output-dir",
        default="./data",
        help="Output directory (default: ./data)"
    )
    
    parser.add_argument(
        "--list-sectors",
        action="store_true",
        help="List available sectors and exit"
    )
    
    parser.add_argument(
        "--list-files",
        help="List available files for a sector and exit"
    )
    
    args = parser.parse_args()
    
    downloader = MegaDownloader(output_dir=args.output_dir)
    
    # List sectors
    if args.list_sectors:
        downloader.display_sectors()
        return
    
    # List files
    if args.list_files:
        downloader.display_files(args.list_files)
        return
    
    # Direct download mode
    if args.sector and args.file:
        success = downloader.download_file(args.sector, args.file)
        sys.exit(0 if success else 1)
    
    # Interactive mode
    downloader.run_interactive()


if __name__ == "__main__":
    main()
