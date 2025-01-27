import subprocess

# result = subprocess.run("which plantri", shell=True, capture_output=True, text=True)
# result = subprocess.run("plantri -h", shell=True, capture_output=True, text=True)
result = subprocess.run("plantri -p -d -f4 -c1 -m2 -E -e9 5", shell=True, capture_output=True)
print(result.stdout)
print(result.stderr)

