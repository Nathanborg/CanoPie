import sys

print("Installing python-docx in-process via pip...")
try:
    from pip._internal.cli.main import main as pip_main
    pip_main(['install', 'python-docx'])
    print("pip install completed successfully.")
except Exception as e:
    print("pip_main exception:", e)

print("Importing build_user_guide_docx...")
import build_user_guide_docx
print("Running build_docx()...")
build_user_guide_docx.build_docx()
