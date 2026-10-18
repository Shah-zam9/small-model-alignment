import subprocess
import re

git_files = subprocess.check_output(['git', 'ls-files'], text=True).splitlines()

patterns = [
    re.compile(r'hf_[a-zA-Z0-9]{34}'),
    re.compile(r'AIza[0-9A-Za-z-_]{35}'),
    re.compile(r'\"key\":\s*\"[a-f0-9]{32}\"'),
]

found = False
for fpath in git_files:
    try:
        with open(fpath, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
            for p in patterns:
                if p.search(content):
                    found = True
                    break
    except Exception:
        pass

log_diff = subprocess.check_output(['git', 'log', '-p', '-n', '20'], text=True, errors='ignore')
for p in patterns:
    if p.search(log_diff):
        found = True
        break

if found:
    print('SECRETS_FOUND: YES')
else:
    print('SECRETS_FOUND: NO')
