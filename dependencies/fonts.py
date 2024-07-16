import os
import platform
import shutil

def install_fonts():
    # Define the fonts directory
    fonts_dir = 'dependencies/fonts'
    
    # Check if the fonts directory exists
    if not os.path.exists(fonts_dir):
        print(f"Directory {fonts_dir} does not exist.")
        return

    # Get a list of all .ttf and .otf font files in the fonts directory
    font_files = [f for f in os.listdir(fonts_dir) if f.endswith('.ttf') or f.endswith('.otf')]

    if not font_files:
        print("No .ttf or .otf font files found in the directory.")
        return


    # Detect the operating system
    system = platform.system()

    if system == 'Windows':
        # Windows font installation
        fonts_dest_dir = os.path.join(os.environ['WINDIR'], 'Fonts')
        for font in font_files:
            src_font_path = os.path.join(fonts_dir, font)
            dest_font_path = os.path.join(fonts_dest_dir, font)
            print(f"Installing {font} to {fonts_dest_dir}")
            shutil.copy(src_font_path, dest_font_path)
            # Register the font
            os.system(f'REG ADD "HKLM\\SOFTWARE\\Microsoft\\Windows NT\\CurrentVersion\\Fonts" /v "{font}" /t REG_SZ /d "{font}" /f')
        print("Fonts installed on Windows.")

    elif system == 'Darwin':
        # macOS font installation
        fonts_dest_dir = os.path.expanduser('~/Library/Fonts')
        for font in font_files:
            src_font_path = os.path.join(fonts_dir, font)
            dest_font_path = os.path.join(fonts_dest_dir, font)
            print(f"Installing {font} to {fonts_dest_dir}")
            shutil.copy(src_font_path, dest_font_path)
        print("Fonts installed on macOS.")

    else:
        print(f"Unsupported operating system: {system}")

def font_exists(font_name):
    system = platform.system()
    print(system)
    if system == 'Windows':
        fonts_dest_dir = os.path.join(os.environ['WINDIR'], 'dependencies/fonts')
        return os.path.exists(os.path.join(fonts_dest_dir, font_name))
    elif system == 'Darwin' or system == 'Linux':
        fonts_dest_dir = os.path.join(os.getcwd(), 'dependencies/fonts')
        print(fonts_dest_dir, '---------------------------')
        return os.path.exists(os.path.join(fonts_dest_dir, font_name))
    else:
        return False
