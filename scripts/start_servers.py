"""
Script to start both frontend and backend servers for testing the dashboard.
This script manages both processes and provides a unified interface.
"""

import subprocess
import sys
import time
import signal
import os
from pathlib import Path
import webbrowser
from threading import Timer

# Global process references
backend_process = None
frontend_process = None


def print_header(text):
    """Print a formatted header."""
    print("\n" + "=" * 80)
    print(f"  {text}")
    print("=" * 80 + "\n")


def check_dependencies():
    """Check if required dependencies are installed."""
    print_header("CHECKING DEPENDENCIES")
    
    issues = []
    
    # Check Python packages
    try:
        import fastapi
        import uvicorn
        print("✓ FastAPI and Uvicorn installed")
    except ImportError:
        issues.append("FastAPI/Uvicorn not installed. Run: pip install fastapi uvicorn")
    
    # Check Node.js
    try:
        result = subprocess.run(['node', '--version'], 
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            print(f"✓ Node.js installed: {result.stdout.strip()}")
        else:
            issues.append("Node.js not found. Please install Node.js.")
    except (subprocess.TimeoutExpired, FileNotFoundError):
        issues.append("Node.js not found. Please install Node.js.")
    
    # Check npm
    try:
        result = subprocess.run(['npm', '--version'], 
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            print(f"✓ npm installed: {result.returncode}")
        else:
            issues.append("npm not found. Please install npm.")
    except (subprocess.TimeoutExpired, FileNotFoundError):
        issues.append("npm not found. Please install npm.")
    
    if issues:
        print("\n⚠️  Issues found:")
        for issue in issues:
            print(f"  - {issue}")
        return False
    
    print("\n✓ All dependencies are installed!")
    return True


def check_model_files():
    """Check if model files exist."""
    print_header("CHECKING MODEL FILES")
    
    issues = []
    
    # Check Solution 1 models
    ml_models_dir = Path("solution_1_ml/outputs/models")
    if not ml_models_dir.exists():
        issues.append(f"Solution 1 models directory not found: {ml_models_dir}")
        print(f"⚠️  {issues[-1]}")
    else:
        model_files = list(ml_models_dir.glob("*.pkl")) + list(ml_models_dir.glob("*.pth"))
        if not model_files:
            issues.append("Solution 1 model files not found. Run training first.")
            print(f"⚠️  {issues[-1]}")
        else:
            print(f"✓ Found {len(model_files)} Solution 1 model files")
    
    # Check Solution 2 models
    dl_model_file = Path("solution_2_dl/outputs/dl_models/best_model.pt")
    if not dl_model_file.exists():
        issues.append(f"Solution 2 model not found: {dl_model_file}")
        print(f"⚠️  {issues[-1]}")
        print("   Run: python solution_2_dl/train_dl_model.py")
    else:
        print(f"✓ Solution 2 model found: {dl_model_file}")
    
    if issues:
        print("\n⚠️  Some model files are missing.")
        print("   The servers will start, but some features may not work.")
        print("   Run 'python scripts/generate_all_results.py' to train models.")
        return False
    
    print("\n✓ All model files are present!")
    return True


def start_backend_server(port=8000, solution='both'):
    """Start the backend FastAPI server."""
    global backend_process
    
    print_header("STARTING BACKEND SERVER")
    
    # Determine which server to start
    if solution == 'solution1' or solution == 'both':
        server_file = Path("testing/api/server.py")
    elif solution == 'solution2':
        server_file = Path("testing/api/server_v2.py")
    else:
        print(f"❌ Unknown solution: {solution}")
        return False
    
    if not server_file.exists():
        print(f"❌ Server file not found: {server_file}")
        return False
    
    print(f"Starting backend server: {server_file.name}")
    print(f"Port: {port}")
    print(f"Solution: {solution}")
    
    try:
        # Change to testing directory for proper imports
        testing_dir = Path("testing")
        os.chdir(testing_dir)
        
        # Start uvicorn server
        backend_process = subprocess.Popen(
            [sys.executable, "-m", "uvicorn", f"api.{server_file.stem}:app", 
             "--host", "0.0.0.0", "--port", str(port), "--reload"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
        
        # Return to original directory
        os.chdir("..")
        
        # Wait a bit to see if server starts successfully
        time.sleep(3)
        
        if backend_process.poll() is None:
            print(f"✓ Backend server started successfully!")
            print(f"  API available at: http://localhost:{port}")
            print(f"  API docs at: http://localhost:{port}/docs")
            return True
        else:
            print("❌ Backend server failed to start")
            return False
            
    except Exception as e:
        print(f"❌ Error starting backend server: {e}")
        return False


def start_frontend_server(port=3000):
    """Start the frontend Next.js server."""
    global frontend_process
    
    print_header("STARTING FRONTEND SERVER")
    
    dashboard_dir = Path("testing/dashboard")
    
    if not dashboard_dir.exists():
        print(f"❌ Dashboard directory not found: {dashboard_dir}")
        return False
    
    # Check if node_modules exists
    node_modules = dashboard_dir / "node_modules"
    if not node_modules.exists():
        print("⚠️  node_modules not found. Installing dependencies...")
        print("   This may take a few minutes...")
        
        try:
            os.chdir(dashboard_dir)
            result = subprocess.run(
                ['npm', 'install'],
                capture_output=True,
                text=True,
                timeout=300  # 5 minutes timeout
            )
            
            if result.returncode != 0:
                print(f"❌ npm install failed: {result.stderr}")
                os.chdir("..")
                return False
            
            print("✓ Dependencies installed successfully!")
            os.chdir("..")
            
        except subprocess.TimeoutExpired:
            print("❌ npm install timed out")
            os.chdir("..")
            return False
        except Exception as e:
            print(f"❌ Error installing dependencies: {e}")
            os.chdir("..")
            return False
    
    print(f"Starting frontend server on port {port}...")
    
    try:
        os.chdir(dashboard_dir)
        
        # Start Next.js dev server
        frontend_process = subprocess.Popen(
            ['npm', 'run', 'dev'],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
        
        # Return to original directory
        os.chdir("../..")
        
        # Wait a bit to see if server starts successfully
        time.sleep(5)
        
        if frontend_process.poll() is None:
            print(f"✓ Frontend server started successfully!")
            print(f"  Dashboard available at: http://localhost:{port}")
            return True
        else:
            print("❌ Frontend server failed to start")
            return False
            
    except Exception as e:
        print(f"❌ Error starting frontend server: {e}")
        os.chdir("../..")
        return False


def open_browser(url, delay=3):
    """Open browser after a delay."""
    def open_url():
        webbrowser.open(url)
    
    Timer(delay, open_url).start()


def cleanup_processes():
    """Clean up all running processes."""
    global backend_process, frontend_process
    
    print_header("SHUTTING DOWN SERVERS")
    
    if backend_process:
        print("Stopping backend server...")
        try:
            backend_process.terminate()
            backend_process.wait(timeout=5)
            print("✓ Backend server stopped")
        except subprocess.TimeoutExpired:
            backend_process.kill()
            print("✓ Backend server force stopped")
        except Exception as e:
            print(f"⚠️  Error stopping backend: {e}")
    
    if frontend_process:
        print("Stopping frontend server...")
        try:
            frontend_process.terminate()
            frontend_process.wait(timeout=5)
            print("✓ Frontend server stopped")
        except subprocess.TimeoutExpired:
            frontend_process.kill()
            print("✓ Frontend server force stopped")
        except Exception as e:
            print(f"⚠️  Error stopping frontend: {e}")


def signal_handler(sig, frame):
    """Handle interrupt signals."""
    print("\n\n⚠️  Interrupt received. Shutting down...")
    cleanup_processes()
    sys.exit(0)


def main():
    """Main execution function."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Start both frontend and backend servers for the dashboard'
    )
    parser.add_argument('--backend-port', type=int, default=8000,
                       help='Backend server port (default: 8000)')
    parser.add_argument('--frontend-port', type=int, default=3000,
                       help='Frontend server port (default: 3000)')
    parser.add_argument('--solution', type=str, default='both',
                       choices=['solution1', 'solution2', 'both'],
                       help='Which solution to run (default: both)')
    parser.add_argument('--skip-checks', action='store_true',
                       help='Skip dependency and model file checks')
    parser.add_argument('--no-browser', action='store_true',
                       help='Do not open browser automatically')
    
    args = parser.parse_args()
    
    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    print_header("DASHBOARD SERVER STARTUP")
    print("This script will start both the backend API and frontend dashboard.")
    print("\nPress Ctrl+C to stop both servers.\n")
    
    # Check dependencies
    if not args.skip_checks:
        if not check_dependencies():
            print("\n⚠️  Dependency check failed. Continuing anyway...")
            response = input("Continue? (y/n): ")
            if response.lower() != 'y':
                return 1
        
        # Check model files (non-blocking)
        check_model_files()
    
    # Start backend
    if not start_backend_server(args.backend_port, args.solution):
        print("\n❌ Failed to start backend server")
        return 1
    
    # Start frontend
    if not start_frontend_server(args.frontend_port):
        print("\n❌ Failed to start frontend server")
        cleanup_processes()
        return 1
    
    # Open browser
    if not args.no_browser:
        print("\nOpening browser in 3 seconds...")
        open_browser(f"http://localhost:{args.frontend_port}")
    
    print_header("SERVERS RUNNING")
    print("✓ Backend API: http://localhost:{}".format(args.backend_port))
    print("✓ Frontend Dashboard: http://localhost:{}".format(args.frontend_port))
    print("\nPress Ctrl+C to stop both servers.\n")
    
    # Monitor processes
    try:
        while True:
            # Check if processes are still running
            if backend_process and backend_process.poll() is not None:
                print("\n⚠️  Backend server stopped unexpectedly")
                break
            
            if frontend_process and frontend_process.poll() is not None:
                print("\n⚠️  Frontend server stopped unexpectedly")
                break
            
            time.sleep(1)
            
    except KeyboardInterrupt:
        pass
    
    cleanup_processes()
    return 0


if __name__ == "__main__":
    exit(main())
