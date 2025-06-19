#!/usr/bin/env python3
"""
FIXED: One-click setup script for Bubble Dynamics Web App
Automatically sets up and runs the Streamlit web application
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path


def check_python_version():
    """Check if Python version is compatible - FIXED VERSION"""
    version_info = sys.version_info
    print(f"🐍 Python version: {version_info.major}.{version_info.minor}.{version_info.micro}")

    if version_info < (3, 7, 0):
        print("❌ Python 3.7 or higher is required")
        print(f"   Current version: {version_info.major}.{version_info.minor}.{version_info.micro}")
        return False

    print(f"✅ Python version is compatible: {version_info.major}.{version_info.minor}.{version_info.micro}")
    return True


def install_requirements():
    """Install required packages"""
    requirements = [
        "streamlit>=1.25.0",
        "numpy>=1.21.0",
        "matplotlib>=3.5.0",
        "scipy>=1.7.0",
        "pillow>=8.0.0",
        "pandas>=1.3.0"
    ]

    # Ask about TensorFlow
    print("\n🤖 TensorFlow is needed for ML prediction features")
    try_tensorflow = input("Install TensorFlow for ML features? (y/n) [y]: ").lower()
    if try_tensorflow == '' or try_tensorflow.startswith('y'):
        requirements.append("tensorflow>=2.8.0")
        print("   📦 TensorFlow will be installed")
    else:
        print("   ⚠️  ML features will be disabled without TensorFlow")

    print("\n📦 Installing requirements...")
    success_count = 0

    for req in requirements:
        try:
            print(f"   📥 Installing {req.split('>=')[0]}...")
            result = subprocess.run([sys.executable, "-m", "pip", "install", req],
                                    capture_output=True, text=True, timeout=300)

            if result.returncode == 0:
                print(f"   ✅ {req.split('>=')[0]} installed successfully")
                success_count += 1
            else:
                print(f"   ⚠️  Warning: Failed to install {req}")
                print(f"      Error: {result.stderr.strip()}")

        except subprocess.TimeoutExpired:
            print(f"   ⏰ Timeout installing {req}")
        except Exception as e:
            print(f"   ❌ Error installing {req}: {str(e)}")

    print(f"\n✅ Installation completed: {success_count}/{len(requirements)} packages installed")
    return success_count > 0


def create_project_structure():
    """Create necessary files and folders"""
    print("\n📁 Creating project structure...")

    # Create .streamlit directory for config
    streamlit_dir = Path(".streamlit")
    streamlit_dir.mkdir(exist_ok=True)

    # Create config file
    config_content = """[global]
developmentMode = false

[server]
runOnSave = true
port = 8501
maxUploadSize = 200

[browser]
gatherUsageStats = false

[theme]
primaryColor = "#1e88e5"
backgroundColor = "#ffffff"
secondaryBackgroundColor = "#f0f2f6"
textColor = "#262730"
"""

    with open(streamlit_dir / "config.toml", "w") as f:
        f.write(config_content)

    print("   ✅ Streamlit configuration created")
    print("   ✅ Project structure completed")


def check_streamlit_app():
    """Check if main Streamlit app exists"""
    app_file = "streamlit_bubble_app.py"
    if not os.path.exists(app_file):
        print(f"❌ {app_file} not found in current directory")
        print("   📁 Current directory contents:")
        for file in os.listdir('.'):
            if file.endswith('.py'):
                print(f"      🐍 {file}")
        return False
    print(f"✅ Found {app_file}")
    return True


def test_streamlit_installation():
    """Test if Streamlit is properly installed"""
    try:
        result = subprocess.run([sys.executable, "-m", "streamlit", "--version"],
                                capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            version = result.stdout.strip()
            print(f"✅ Streamlit is ready: {version}")
            return True
        else:
            print("❌ Streamlit installation test failed")
            return False
    except Exception as e:
        print(f"❌ Could not test Streamlit: {e}")
        return False


def run_streamlit_app():
    """Launch the Streamlit application"""
    print("\n🚀 Starting Streamlit application...")
    print("   🌐 Your web app will open in your default browser")
    print("   📍 URL: http://localhost:8501")
    print("   ⏹️  Press Ctrl+C to stop the server")
    print("   📱 The app works on mobile devices too!")
    print("-" * 50)

    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", "streamlit_bubble_app.py"])
    except KeyboardInterrupt:
        print("\n👋 Streamlit app stopped by user")
    except FileNotFoundError:
        print("❌ Streamlit command not found")
        print("💡 Try installing manually: pip install streamlit")
    except Exception as e:
        print(f"❌ Error starting Streamlit: {e}")


def create_quick_start_files():
    """Create helpful files for future use"""
    print("\n📝 Creating quick-start files...")

    # Create requirements.txt
    requirements_content = """# Bubble Dynamics Web App Requirements
streamlit>=1.25.0
numpy>=1.21.0
matplotlib>=3.5.0
scipy>=1.7.0
tensorflow>=2.8.0
pillow>=8.0.0
pandas>=1.3.0
plotly>=5.0.0
"""

    with open("requirements.txt", "w") as f:
        f.write(requirements_content)

    # Create quick start batch file for Windows
    batch_content = """@echo off
echo 🚀 Starting Bubble Dynamics Web App...
echo.
cd /d "%~dp0"
python -m streamlit run streamlit_bubble_app.py
pause
"""

    with open("start_app.bat", "w") as f:
        f.write(batch_content)

    # Create quick start shell script for Linux/Mac
    shell_content = """#!/bin/bash
echo "🚀 Starting Bubble Dynamics Web App..."
echo
cd "$(dirname "$0")"
python -m streamlit run streamlit_bubble_app.py
"""

    with open("start_app.sh", "w") as f:
        f.write(shell_content)

    # Make shell script executable (on Unix systems)
    try:
        os.chmod("start_app.sh", 0o755)
    except:
        pass  # Ignore on Windows

    print("   ✅ requirements.txt created")
    print("   ✅ start_app.bat created (Windows)")
    print("   ✅ start_app.sh created (Linux/Mac)")


def main():
    """Main setup process - FIXED VERSION"""
    print("🫧 Bubble Dynamics Web App Setup (Fixed)")
    print("=" * 45)

    # Check system requirements
    if not check_python_version():
        input("Press Enter to exit...")
        return

    # Check if app file exists
    if not check_streamlit_app():
        input("Press Enter to exit...")
        return

    print("\n🛠️  Setting up your web application...")

    # Install requirements
    if not install_requirements():
        print("⚠️  Some packages failed to install, but continuing...")

    # Test Streamlit installation
    if not test_streamlit_installation():
        print("❌ Streamlit installation failed")
        print("💡 Try manual installation: pip install streamlit")
        input("Press Enter to exit...")
        return

    # Create project structure
    create_project_structure()

    # Create helpful files
    create_quick_start_files()

    print("\n🎉 Setup completed successfully!")
    print("\n" + "=" * 45)
    print("📋 What's Ready:")
    print("   ✅ Web application configured")
    print("   ✅ Dependencies installed")
    print("   ✅ Quick-start files created")
    print("\n📋 Next Steps:")
    print("   1. 🚀 Test locally (run now)")
    print("   2. 🌐 Deploy to Streamlit Cloud (free)")
    print("   3. 📤 Share with colleagues")
    print("=" * 45)

    # Ask if user wants to run the app now
    print("\n🎯 Ready to launch your web app!")
    run_now = input("🚀 Start the web application now? (y/n) [y]: ").lower()

    if run_now == '' or run_now.startswith('y'):
        run_streamlit_app()
    else:
        print("\n💡 To run later:")
        print("   • Double-click: start_app.bat (Windows)")
        print("   • Or run: streamlit run streamlit_bubble_app.py")
        print("   • Or run: python setup_web_app_fixed.py")


if __name__ == "__main__":
    main()