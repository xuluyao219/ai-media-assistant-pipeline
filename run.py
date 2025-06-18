#!/usr/bin/env python
"""
Start Web Application
"""

from web_app import app

if __name__ == '__main__':
    print("Starting Multimodal AI Media Analysis System")
    print("Access URL: http://localhost:5000")
    print("Press Ctrl+C to stop the server")
    print("="*60)
    
    # Run application
    app.run(host='0.0.0.0', port=5000, debug=False)
