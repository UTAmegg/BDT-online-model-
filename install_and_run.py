#!/usr/bin/env python3
"""
Emergency installer - bypasses pip issues by using different methods
"""

import sys
import subprocess
import os


def find_correct_python():
    """Find the correct Python executable"""
    print("🔍 Finding correct Python installation...")

    # Try different Python commands
    python_commands = [
        'py',  # Windows Python Launcher
        'python3',  # Standard Python 3
        'python',  # Generic Python
    ]

    for cmd in python_commands:
        try:
            result = subprocess.run([cmd, '--version'], capture_output=True, text=True)
            if result.returncode == 0 and 'Inkscape' not in result.stdout:
                print(f"✅ Found good Python: {cmd}")
                print(f"   Version: {result.stdout.strip()}")

                # Test if pip works with this Python
                pip_result = subprocess.run([cmd, '-m', 'pip', '--version'],
                                            capture_output=True, text=True)
                if pip_result.returncode == 0:
                    print(f"✅ Pip works with {cmd}")
                    return cmd
                else:
                    print(f"❌ Pip doesn't work with {cmd}")

        except FileNotFoundError:
            continue

    print("❌ Could not find a working Python with pip")
    return None


def install_packages(python_cmd):
    """Install required packages using the correct Python"""
    packages = [
        'streamlit',
        'numpy',
        'matplotlib',
        'scipy',
        'pillow',
        'pandas'
    ]

    print(f"\n📦 Installing packages using {python_cmd}...")

    for package in packages:
        print(f"   📥 Installing {package}...")
        try:
            result = subprocess.run([python_cmd, '-m', 'pip', 'install', package],
                                    capture_output=True, text=True, timeout=120)
            if result.returncode == 0:
                print(f"   ✅ {package} installed successfully")
            else:
                print(f"   ⚠️ Failed to install {package}")
                print(f"      Error: {result.stderr.strip()}")
        except subprocess.TimeoutExpired:
            print(f"   ⏰ Timeout installing {package}")
        except Exception as e:
            print(f"   ❌ Error installing {package}: {e}")


def run_streamlit(python_cmd):
    """Run Streamlit with the correct Python"""
    print(f"\n🚀 Starting Streamlit with {python_cmd}...")

    if not os.path.exists('streamlit_bubble_app.py'):
        print("❌ streamlit_bubble_app.py not found")
        return False

    try:
        # Test Streamlit installation first
        test_result = subprocess.run([python_cmd, '-m', 'streamlit', '--version'],
                                     capture_output=True, text=True)
        if test_result.returncode != 0:
            print("❌ Streamlit not properly installed")
            return False

        print("✅ Streamlit is ready")
        print("🌐 Opening web browser...")
        print("📍 URL: http://localhost:8501")
        print("⏹️ Press Ctrl+C to stop")

        # Run Streamlit
        subprocess.run([python_cmd, '-m', 'streamlit', 'run', 'streamlit_bubble_app.py'])
        return True

    except KeyboardInterrupt:
        print("\n👋 Streamlit stopped by user")
        return True
    except Exception as e:
        print(f"❌ Error running Streamlit: {e}")
        return False


def main():
    print("🫧 Emergency Bubble Dynamics Web App Installer")
    print("=" * 50)
    print("🎯 This script finds the correct Python and bypasses Inkscape's Python")
    print()

    # Find correct Python
    python_cmd = find_correct_python()
    if not python_cmd:
        print("\n💡 Manual Solutions:")
        print("1. Open regular Command Prompt (not PyCharm)")
        print("2. Try: py -m pip install streamlit numpy matplotlib scipy")
        print("3. Then: py -m streamlit run streamlit_bubble_app.py")
        print("\n4. Or fix PyCharm Python interpreter:")
        print("   Settings → Project → Python Interpreter → Add System Interpreter")
        input("\nPress Enter to exit...")
        return

    # Install packages
    install_packages(python_cmd)

    # Ask if user wants to run Streamlit
    print("\n🎉 Installation completed!")
    run_now = input("\n🚀 Start the web app now? (y/n) [y]: ").lower()

    if run_now == '' or run_now.startswith('y'):
        success = run_streamlit(python_cmd)
        if not success:
            print(f"\n💡 To run manually later:")
            print(f"   {python_cmd} -m streamlit run streamlit_bubble_app.py")
    else:
        print(f"\n💡 To run later:")
        print(f"   {python_cmd} -m streamlit run streamlit_bubble_app.py")


if __name__ == "__main__":
    main()