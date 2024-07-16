import subprocess
import platform

def is_imagemagick_installed():
    """Check if ImageMagick is installed."""
    command = "magick --version" if platform.system() == "Windows" else "convert --version"
    try:
        subprocess.run(command, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return True
    except subprocess.CalledProcessError:
        return False

def install_imagemagick():
    """Prompt the user to install ImageMagick."""
    print("ImageMagick is required but not found on your system.")
    if platform.system() == "Windows":
        print("Please download and install ImageMagick from: https://imagemagick.org/script/download.php")
    else:
        print("You can install ImageMagick using your package manager (e.g., 'brew install imagemagick' on macOS)")

